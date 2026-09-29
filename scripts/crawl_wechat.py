#!/usr/bin/env python
r"""微信公众号文章采集（经搜狗微信搜索）。

为什么需要这个脚本
------------------
13 个学院/部门站点（xgb / zhaosheng / phy / chem / mse / cie / mee / cee /
fl / lsxy / bs / gggl / marx）对本机所在网络返回 403，已确认是**按来源网络
放行**、与认证无关（见 README 8.1）。但同样内容大量存在于各学院的微信
公众号里——教师介绍、专业介绍、招聘公告、学院动态，覆盖面甚至比官网更全。

本脚本走「搜狗微信搜索 → 解析 JS 跳转 → 抓 mp.weixin.qq.com 正文」，
原始 HTML 落盘到 `data/raw_html/`（文件名 `wechat_*`，与 crawl.py 的缓存
格式一致），随后 `extract_texts.py` + `build_index.py` 可直接消费，
不需要改任何下游代码。

用法::

    uv run python scripts/crawl_wechat.py --dry-run          # 只搜索不抓正文
    uv run python scripts/crawl_wechat.py                     # 用默认关键词全量抓
    uv run python scripts/crawl_wechat.py --max-per-keyword 30
    uv run python scripts/crawl_wechat.py --keywords "湘潭大学 化学学院,湘潭大学 材料学院"

注意
----
* 搜狗的 `/link?url=...` 是**临时**跳转链，几小时后失效；本脚本是"搜到即抓"，
  不做链接持久化，只持久化抓到的正文。
* 有速率限制，连续高频请求会触发验证码。脚本内置 `--delay` 与验证码熔断，
  触发后会**立即停止**并保留已抓内容。
* 抓到的正文质量取决于文章本身，公众号文章常带推广、排版噪声，
  但比拿不到内容好得多。
"""

import argparse
import hashlib
import html as _html
import json
import logging
import random
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.crawler.spider import Spider  # noqa: E402

logger = logging.getLogger("crawl_wechat")

SEARCH_URL = "https://weixin.sogou.com/weixin"
HOME_URL = "https://weixin.sogou.com/"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 被 WAF 403 挡住的单位 + 通用主题词。这些是官网拿不到、只能靠公众号补的内容。
DEFAULT_KEYWORDS = [
    "湘潭大学 学生工作部",
    "湘潭大学 招生",
    "湘潭大学 物理与光电工程学院",
    "湘潭大学 化学学院",
    "湘潭大学 材料科学与工程学院",
    "湘潭大学 计算机学院",
    "湘潭大学 机械工程与力学学院",
    "湘潭大学 土木工程学院",
    "湘潭大学 外国语学院",
    "湘潭大学 历史文化学院",
    "湘潭大学 商学院",
    "湘潭大学 公共管理学院",
    "湘潭大学 马克思主义学院",
    "湘潭大学 图书馆",
    "湘潭大学 专业介绍",
    "湘潭大学 双一流",
    "湘潭大学 就业",
    "湘潭大学 导师",
]

# 正文过短的多半是排版空壳或"内容已删除"占位页
MIN_CHARS = 200

# 命中这些说明被反爬拦了，继续请求没有意义
_BLOCK_MARKERS = (
    "请输入验证码",
    "antispider",
    "您的访问过于频繁",
    "环境异常",
)


class AntiSpiderError(RuntimeError):
    """搜狗/微信反爬拦截。"""


@dataclass
class Article:
    title: str
    account: str
    date: str
    sogou_url: str
    snippet: str = ""
    real_url: str = ""
    file: str = ""
    chars: int = 0
    keyword: str = ""
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

_LI_RE = re.compile(r'<li id="sogou_vr_\d+_box_\d+"(.*?)</li>', re.S)
_TITLE_RE = re.compile(r"<h3>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S)
# 公众号名和日期都在 <div class="s-p"> 里，按**结构**取而不按 class 取：
# 公众号 span 的 class 会变（实测 all-time-y2），并非教程里说的 <a class="account">。
_SP_RE = re.compile(r'<div class="s-p">(.*?)</div>', re.S)
_SPAN_RE = re.compile(r"<span[^>]*>(.*?)</span>", re.S)
_ACCOUNT_FALLBACK_RE = re.compile(r'<a[^>]*class="account"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'<p class="txt-info"[^>]*>(.*?)</p>', re.S)
# 新条目日期由 JS 写入：document.write(timeConvert('1772897135'))
_TS_RE = re.compile(r"timeConvert\('(\d{9,11})'\)")
_TS_ATTR_RE = re.compile(r'\st="(\d{10})"')
# 旧条目直接给文本日期（2024-04-03 / 2024年4月3日）
_DATE_LIKE_RE = re.compile(r"\d{4}\s*[-/年]")
# 搜狗 link 页把真实地址拆成多段 url += '...' 拼出来，用来绕开静态解析
_JS_FRAG_RE = re.compile(r"url\s*\+=\s*'([^']*)'")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _plain(fragment: str) -> str:
    """HTML 片段 → 纯文本。"""
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub("", fragment))).strip()


def _fmt_date(ts: str) -> str:
    """unix 时间戳 → YYYY-MM-DD；失败则原样返回。"""
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return ts


def parse_account_and_date(block: str) -> tuple[str, str]:
    """从搜索结果的 `<div class="s-p">` 里取 (公众号名, 日期)。

    两种日期格式都要认：新条目是 `timeConvert('时间戳')` 由 JS 写入，
    旧条目直接是 `2024-04-03` 文本。认不出日期不算错，`extract_texts`
    只用文件名和正文，日期仅用于清单里的人工核对。
    """
    sp = _SP_RE.search(block)
    if not sp:
        acc = _ACCOUNT_FALLBACK_RE.search(block)
        ts = _TS_ATTR_RE.search(block)
        return (
            _plain(acc.group(1)) if acc else "",
            _fmt_date(ts.group(1)) if ts else "",
        )

    account = date = ""
    for frag in _SPAN_RE.findall(sp.group(1)):
        ts = _TS_RE.search(frag)
        if ts:
            date = date or _fmt_date(ts.group(1))
            continue
        txt = _plain(frag)
        if not txt:
            continue
        if _DATE_LIKE_RE.search(txt):
            date = date or txt
        elif not account:
            account = txt

    if not date:
        ts = _TS_ATTR_RE.search(block)
        if ts:
            date = _fmt_date(ts.group(1))
    return account, date


def parse_search_results(html: str, keyword: str) -> list[Article]:
    """解析搜狗微信搜索结果页。"""
    out: list[Article] = []
    for block in _LI_RE.findall(html):
        m = _TITLE_RE.search(block)
        if not m:
            continue
        href = _html.unescape(m.group(1))
        if href.startswith("/"):
            href = "https://weixin.sogou.com" + href
        account, date = parse_account_and_date(block)
        snip = _SNIPPET_RE.search(block)
        out.append(
            Article(
                title=_plain(m.group(2)),
                account=account,
                date=date,
                sogou_url=href,
                snippet=_plain(snip.group(1)) if snip else "",
                keyword=keyword,
            )
        )
    return out


def resolve_real_url(link_page_html: str) -> str | None:
    """从搜狗 link 页的 JS 拼接片段还原出 mp.weixin.qq.com 真实地址。"""
    frags = _JS_FRAG_RE.findall(link_page_html)
    if not frags:
        return None
    url = "".join(frags).replace("@", "").strip()
    if not url.startswith("http"):
        return None
    # 防御：拼接片段里可能残留 JS 转义
    return url.replace("\\/", "/")


_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.S | re.I)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def compact_html(raw: bytes) -> bytes:
    """落盘前剥掉 script/style/注释。

    微信文章页实测 3.5MB，而正文只有几 KB——其余全是内联脚本、样式和
    base64 占位。正文位于 `#js_content` 的**静态** HTML 中，删掉 script
    不影响后续重抽，但能把单页压到 1/30，几百篇的缓存才不会涨到 GB 级。
    非 UTF-8 页面原样返回，避免误伤编码。
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    text = _SCRIPT_RE.sub("", text)
    text = _STYLE_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    return text.encode("utf-8")


def _check_block(text: str) -> None:
    for marker in _BLOCK_MARKERS:
        if marker in text:
            raise AntiSpiderError(f"命中反爬标记：{marker}")


# --------------------------------------------------------------------------- #
# 文件名
# --------------------------------------------------------------------------- #

_SLUG_KEEP_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def _slug(text: str, limit: int = 48) -> str:
    """标题 → 文件名安全片段（保留中文，便于人工核对）。"""
    s = _SLUG_KEEP_RE.sub("_", text).strip("_")
    return s[:limit] or "untitled"


def _file_name(article: Article) -> str:
    """`wechat_{公众号}_{标题}_{hash8}.html`。

    spider._safe_name 只用 netloc+path，而搜狗链接与微信临时链接的参数
    全在 query 里（path 都是 /link 或 /s），直接用会全部撞名，所以自建命名。
    末尾附 hash 保证同名不同文也能区分。
    """
    digest = hashlib.blake2b(article.sogou_url.encode("utf-8"), digest_size=4).hexdigest()
    acc = _slug(article.account, 16) if article.account else "unknown"
    return f"wechat_{acc}_{_slug(article.title)}_{digest}.html"


# --------------------------------------------------------------------------- #
# 采集器
# --------------------------------------------------------------------------- #


class WeChatCrawler:
    def __init__(
        self,
        delay: float = 3.0,
        search_delay: float = 5.0,
        timeout: int = 25,
        html_dir: Path | None = None,
        manifest: Path | None = None,
        compact: bool = True,
    ):
        self.delay = delay
        self.search_delay = search_delay
        self.timeout = timeout
        self.html_dir = html_dir
        self.manifest = manifest
        self.compact = compact
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Referer": HOME_URL,
            }
        )
        # 只用离线抽取逻辑，不联网
        self._extractor = Spider(prefer_ipv4=False)
        self._warmed = False
        self._seen_urls: set[str] = set()
        self._seen_titles: set[str] = set()
        self.stats = {"搜索": 0, "命中": 0, "成功": 0, "跳过重复": 0, "失败": 0, "过短": 0}

    # ---------- 基础请求 ----------

    def _get(self, url: str, **kw) -> requests.Response:
        kw.setdefault("timeout", self.timeout)
        kw.setdefault("allow_redirects", True)
        resp = self.session.get(url, **kw)
        if resp.status_code == 200:
            _check_block(resp.text[:20000])
        return resp

    def warmup(self) -> None:
        """先访问首页拿 cookie，否则搜索页会直接给验证码。"""
        if self._warmed:
            return
        r = self.session.get(HOME_URL, timeout=self.timeout)
        logger.debug("预热 %s → %s，cookies=%s", HOME_URL, r.status_code, list(self.session.cookies.keys()))
        self._warmed = True
        time.sleep(1.0)

    # ---------- 搜索 ----------

    def search(self, keyword: str, page: int = 1) -> list[Article]:
        self.warmup()
        params = {"type": 2, "query": keyword, "ie": "utf8"}
        if page > 1:
            params["page"] = page
        url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
        r = self._get(url, headers={"Referer": HOME_URL})
        self.stats["搜索"] += 1
        if r.status_code != 200:
            logger.warning("搜索失败 [%s] p%d → HTTP %s", keyword, page, r.status_code)
            return []
        items = parse_search_results(r.text, keyword)
        logger.debug("搜索 [%s] p%d → %d 条", keyword, page, len(items))
        return items

    # ---------- 抓正文 ----------

    def _resolve(self, sogou_url: str) -> str | None:
        # 搜狗 link 页必须带搜狗站内 Referer
        r = self._get(sogou_url, headers={"Referer": SEARCH_URL})
        if r.status_code != 200:
            return None
        return resolve_real_url(r.text)

    def _fetch_article_html(self, real_url: str) -> bytes | None:
        headers = {
            "Referer": "https://weixin.sogou.com/",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        r = self.session.get(real_url, timeout=self.timeout, headers=headers)
        if r.status_code != 200:
            return None
        raw = r.content
        text = self._extractor._text_from_bytes(raw)
        if not text or len(text) <= MIN_CHARS:
            return None
        return raw

    def grab(self, article: Article, dry_run: bool = False) -> bool:
        """抓一篇。返回是否成功落盘。"""
        key = article.title.strip()
        if key and key in self._seen_titles:
            self.stats["跳过重复"] += 1
            return False
        if article.sogou_url in self._seen_urls:
            self.stats["跳过重复"] += 1
            return False
        self._seen_urls.add(article.sogou_url)
        if key:
            self._seen_titles.add(key)

        if dry_run:
            logger.info("  [dry-run] %s | %s", article.account or "?", article.title[:50])
            return False

        try:
            real = self._resolve(article.sogou_url)
        except AntiSpiderError:
            raise
        except requests.RequestException as exc:
            logger.debug("解析跳转失败：%s", exc)
            real = None

        if not real:
            self.stats["失败"] += 1
            logger.debug("  跳过（无法解析真实地址）：%s", article.title[:40])
            return False

        article.real_url = real
        try:
            raw = self._fetch_article_html(real)
        except AntiSpiderError:
            raise
        except requests.RequestException as exc:
            self.stats["失败"] += 1
            logger.debug("  抓取失败：%s | %s", exc, article.title[:40])
            return False

        if raw is None:
            self.stats["过短"] += 1
            logger.debug("  跳过（正文过短/已删除）：%s", article.title[:40])
            return False

        article.file = _file_name(article)
        article.chars = len(self._extractor._text_from_bytes(raw))
        if self.html_dir:
            if self.compact:
                raw = compact_html(raw)
            self.html_dir.mkdir(parents=True, exist_ok=True)
            (self.html_dir / article.file).write_bytes(raw)
        self.stats["成功"] += 1
        logger.info(
            "  ✓ %5d 字 | %s | %s",
            article.chars,
            article.account or "?",
            article.title[:44],
        )
        self._append_manifest(article)
        return True

    # ---------- 清单（同时充当断点续爬状态） ----------

    def load_manifest(self) -> None:
        if not self.manifest or not self.manifest.exists():
            return
        try:
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("title"):
                    self._seen_titles.add(d["title"].strip())
                if d.get("sogou_url"):
                    self._seen_urls.add(d["sogou_url"])
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("清单读取失败，将当作全新任务：%s", exc)

    def _append_manifest(self, article: Article) -> None:
        if not self.manifest:
            return
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "title": article.title,
            "account": article.account,
            "date": article.date,
            "keyword": article.keyword,
            "sogou_url": article.sogou_url,
            "file": article.file,
            "chars": article.chars,
        }
        with self.manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sleep_between_articles(self) -> None:
        # 加抖动，避免固定间隔被识别
        time.sleep(self.delay + random.uniform(0, self.delay * 0.4))


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="经搜狗微信搜索采集公众号文章")
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="逗号分隔关键词；留空则用内置默认词表（覆盖被 403 的学院/部门）",
    )
    parser.add_argument("--pages", type=int, default=3, help="每个关键词翻几页（每页约 10 条）")
    parser.add_argument("--max-per-keyword", type=int, default=24, help="每个关键词最多抓几篇正文")
    parser.add_argument("--max-total", type=int, default=600, help="全局正文抓取上限")
    parser.add_argument("--delay", type=float, default=3.0, help="正文抓取间隔秒（含抖动）")
    parser.add_argument("--search-delay", type=float, default=6.0, help="搜索翻页间隔秒")
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=_ROOT / "data" / "raw_html",
        help="原始 HTML 落盘目录（默认与 crawl.py 共用，便于统一重抽）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "data" / "wechat" / "manifest.jsonl",
        help="采集清单，同时作为断点续爬状态",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="落盘时保留完整 HTML（默认会剥掉 script/style，单页从 3.5MB 压到约 100KB）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只搜索并列出，不抓正文")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    keywords = (
        [k.strip() for k in args.keywords.split(",") if k.strip()]
        if args.keywords
        else list(DEFAULT_KEYWORDS)
    )

    crawler = WeChatCrawler(
        delay=args.delay,
        search_delay=args.search_delay,
        timeout=args.timeout,
        html_dir=args.html_dir,
        manifest=args.manifest,
        compact=not args.no_compact,
    )
    crawler.load_manifest()
    if crawler._seen_titles:
        logger.info("清单已有 %d 篇，重复的会自动跳过", len(crawler._seen_titles))

    logger.info("关键词 %d 个，每个最多 %d 篇，全局上限 %d 篇", len(keywords), args.max_per_keyword, args.max_total)

    stopped = False
    try:
        for kw in keywords:
            if crawler.stats["成功"] >= args.max_total:
                logger.info("已达全局上限 %d 篇，提前结束", args.max_total)
                break
            logger.info("── 关键词：%s", kw)
            got = 0
            for page in range(1, args.pages + 1):
                try:
                    items = crawler.search(kw, page)
                except AntiSpiderError as exc:
                    logger.warning("搜索被拦（%s），停止。建议加大 --delay 或稍后再试。", exc)
                    stopped = True
                    break
                except requests.RequestException as exc:
                    logger.warning("搜索请求异常：%s", exc)
                    break
                if not items:
                    break
                crawler.stats["命中"] += len(items)
                for art in items:
                    if got >= args.max_per_keyword or crawler.stats["成功"] >= args.max_total:
                        break
                    try:
                        if crawler.grab(art, dry_run=args.dry_run):
                            got += 1
                    except AntiSpiderError as exc:
                        logger.warning("抓取被拦（%s），停止。已抓内容已保留。", exc)
                        stopped = True
                        break
                    if not args.dry_run:
                        crawler.sleep_between_articles()
                if stopped or got >= args.max_per_keyword:
                    break
                if not args.dry_run:
                    time.sleep(args.search_delay)
            logger.info("   本词完成 %d 篇", got)
            if stopped:
                break
    except KeyboardInterrupt:
        logger.warning("手动中断，已抓内容保留在 %s", args.html_dir)

    s = crawler.stats
    logger.info(
        "采集结束：搜索 %d 次，结果 %d 条，落盘 %d 篇，重复跳过 %d，过短 %d，失败 %d%s",
        s["搜索"], s["命中"], s["成功"], s["跳过重复"], s["过短"], s["失败"],
        "（被反爬中断）" if stopped else "",
    )
    if s["成功"]:
        logger.info("下一步：uv run python scripts/extract_texts.py && uv run python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
