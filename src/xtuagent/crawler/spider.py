import json
import re
import socket
import time
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .site_config import CRAWL_RULES

logger = logging.getLogger(__name__)

# 每处理多少个页面把爬取状态（已访问 + 待爬队列）落盘一次
STATE_SAVE_EVERY = 20

# ---------------------------------------------------------------------------
# 正文抽取规则
# ---------------------------------------------------------------------------
# 正文容器选择器：命中即直接取正文，彻底绕开导航/页脚。
# 湘潭大学各站点多为博达（VSB）CMS，正文在 #vsb_content / .v_news_content；
# 其余为常见 CMS 的正文类名，作为兜底。
CONTENT_SELECTORS = [
    "#vsb_content",
    ".v_news_content",
    ".wp_articlecontent",
    ".TRS_Editor",
    ".article-content",
    ".article_content",
    ".news_content",
    # 微信公众号文章正文（公开文章页无需登录即可访问）
    "#js_content",
    # 知乎回答/文章正文
    ".RichText",
    ".Post-RichText",
    # 百度贴吧帖子正文
    ".d_post_content",
    "#content",
    ".content",
    "#zoom",
]

# 永远不属于正文、可直接删除的标签
_DROP_ALWAYS = ("script", "style", "noscript")

# CMS 的间接附件下载入口：URL 中不含真实后缀，只能靠特征串识别。
# 典型：/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=…&wbfileid=…
ATTACHMENT_PATTERNS = [
    r"urltype=news\.downloadattachurl",
    r"/system/_content/download\.jsp",
    r"[?&](wbfileid|fileid|attachid|downloadid)=",
    r"/_upload/",
    r"/uploadfile/",
    r"/attachment/",
]

# 回退路径下丢弃的标签。注意**不能包含 form**：博达 CMS 的正文
# （#vsb_content）嵌套在 <form> 里，删掉 form 会把正文一并删掉。
_DROP_TAGS = (
    "nav",
    "header",
    "footer",
    "aside",
    "iframe",
    "svg",
    "button",
)

# class/id 按非字母数字切词后，命中以下任一「词元」即判定为导航/页脚容器。
# 采用「词元精确匹配」而非子串匹配，避免误伤 consider / navigation 之类命名。
_JUNK_TOKENS = {
    "nav",
    "menu",
    "navbar",
    "topnav",
    "subnav",
    "topbar",
    "header",
    "footer",
    "breadcrumb",
    "crumb",
    "search",
    "login",
    "banner",
    "copyright",
    "friendlink",
    "link",
    "links",
    "share",
    "sidebar",
    "toolbar",
    "dropdown",
    "pagination",
    "pager",
    "advert",
}

_ipv4_forced = False


def force_ipv4() -> None:
    """强制 IPv4 解析。

    教育网站点常同时解析出 IPv6 地址（如 2001:da8::/32），
    在部分网络环境下 IPv6 不可达会导致全部请求超时。
    此函数将进程内 DNS 解析限制为 IPv4。
    """
    global _ipv4_forced
    if _ipv4_forced:
        return
    _ipv4_forced = True

    original_getaddrinfo = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only


class Spider:
    def __init__(
        self,
        max_depth: int = 3,
        max_pages_per_site: int = 80,
        delay: float = 1.0,
        timeout: int = 30,
        allowed_domains: Optional[list[str]] = None,
        file_types: Optional[list[str]] = None,
        prefer_ipv4: bool = True,
        max_total_pages: int = 4000,
        state_file: Optional[str] = None,
        resume: bool = True,
        html_cache_dir: Optional[str] = None,
        render: bool = False,
        cookies: Optional[dict] = None,
    ):
        if prefer_ipv4:
            force_ipv4()
        self.max_depth = max_depth
        self.max_pages_per_site = max_pages_per_site
        self.max_total_pages = max_total_pages
        self.delay = delay
        self.timeout = timeout
        self.allowed_domains = allowed_domains or CRAWL_RULES["allowed_domains"]
        self.file_types = file_types or CRAWL_RULES["file_types"]
        self.priority_patterns = CRAWL_RULES["priority_patterns"]
        self.content_patterns = CRAWL_RULES["content_patterns"]
        self.exclude_patterns = CRAWL_RULES["exclude_patterns"]

        # 断点续爬：把「已访问 URL + 待爬队列」落盘。整站爬取动辄几十分钟，
        # 一旦中断，没有状态就得从头再来（这正是"爬了一半只剩几十篇"的成因）。
        self._state_file = Path(state_file) if state_file else None
        self._visited: set[str] = set()
        self._queued: set[str] = set()
        self._pending: list[tuple[str, int]] = []
        if resume:
            self._load_state()
        # 缓存原始 HTML：抽取规则（正文容器、编码判定）后续必然会调整，
        # 没有缓存就只能整站重爬才能重抽一次。有了缓存，重新生成 data/texts
        # 是纯本地操作，秒级完成。
        self._html_cache_dir = Path(html_cache_dir) if html_cache_dir else None
        # 渲染为可选兜底：JS 单页应用与 403 WAF 站点只有真实浏览器才能拿到内容，
        # 但渲染慢得多，所以默认关闭、按需开启。
        self.render = render
        self._renderer = None
        self._cache_hits = 0
        self._content_saved: set[str] = set()
        self._pages_this_run = 0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        # 需要登录的页面：由使用者从自己浏览器导出 Cookie 存成本地文件后传入。
        # 注意 Cookie 只解决「认证」，解决不了「网络层 403」（见 README 8.1）。
        if cookies:
            self._session.cookies.update(cookies)

    # ---------- 断点续爬状态 ----------

    def _load_state(self) -> None:
        if not self._state_file or not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            self._visited = set(data.get("visited", []))
            # 待爬队列是恢复进度的关键：只存 visited 会导致重新运行时
            # 起点 URL 因"已访问"被直接跳过，队列瞬间见底、什么也爬不到。
            self._pending = [
                (str(item[0]), int(item[1]))
                for item in data.get("pending", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ]
            # 队列去重集必须与 _pending 同步，否则续爬后 pending 中的 URL
            # 会被再次入队，队列迅速膨胀出大量重复项。
            self._queued = {url for url, _ in self._pending}
            logger.info(
                "检测到爬取进度：已访问 %d 个 URL，待爬 %d 个",
                len(self._visited),
                len(self._pending),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("爬取状态文件损坏，忽略并重新开始：%s", exc)
            self._visited = set()
            self._pending = []

    def save_state(self) -> None:
        if not self._state_file:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(
                    {
                        "visited": sorted(self._visited),
                        "pending": [[u, d] for u, d in self._pending],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存爬取状态失败：%s", exc)

    def _is_allowed_domain(self, url: str) -> bool:
        hostname = urlparse(url).hostname
        if not hostname:
            return False
        return any(hostname == d or hostname.endswith("." + d) for d in self.allowed_domains)

    def _is_file_link(self, url: str) -> bool:
        """判断是否为可下载附件。

        除了按后缀名判断，还必须识别 CMS 的**间接下载入口**：博达（VSB）
        把附件挂在 `system/_content/download.jsp?urltype=news.DownloadAttachUrl
        &owner=…&wbfileid=…` 上，**URL 里根本不含 `.pdf`**，只按后缀判断会
        把校内所有附件（培养方案、学生手册、考试安排等）整批漏掉。
        """
        lower = url.lower()
        if any(lower.endswith(ft) for ft in self.file_types):
            return True
        return self._matches_any(lower, ATTACHMENT_PATTERNS)

    def _matches_any(self, url: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, url) for pattern in patterns)

    def _is_excluded(self, url: str) -> bool:
        """命中排除规则的 URL 不入队、不抓取（图片/视频/脚本等）。"""
        return self._matches_any(url, self.exclude_patterns)

    def _is_priority(self, url: str) -> bool:
        """命中优先级规则的 URL 优先入队（正文/通知/培养方案等栏目）。

        附件也按优先级处理：它们的体积/条数远小于页面，却往往是
        「培养方案、考试安排、评奖细则」这类高质量信息的唯一载体，
        不应被普通页面挤掉预算。
        """
        if self._is_file_link(url):
            return True
        return self._matches_any(url, self.priority_patterns)

    def _is_content_page(self, url: str) -> bool:
        return self._matches_any(url, self.content_patterns)

    def _enqueueable(self, link: str, site_counts: dict) -> bool:
        """未访问过、未在队列中、且该站点尚未触及页数上限时才入队。

        `_queued` 去重很关键：同一个 URL 会被很多页面链接到，若不记录
        「已入队」，深度 3 时队列能膨胀到两万条重复项——虽然重复项被弹出时
        会被 `_visited` 挡掉、不消耗页数预算，但会让队列无谓地膨胀、拖慢遍历。
        """
        if link in self._visited or link in self._queued:
            return False
        child_host = urlparse(link).hostname or ""
        return site_counts.get(child_host, 0) < self.max_pages_per_site

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> tuple[list[str], list[str]]:
        """返回 (优先链接, 普通链接)。优先链接会被先访问，保证高价值栏目不被页数上限截断。"""
        priority: list[str] = []
        normal: list[str] = []
        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full_url = urljoin(base_url, href)
            full_url = full_url.split("#")[0]
            if not full_url.startswith("http"):
                continue
            if not self._is_allowed_domain(full_url):
                continue
            if self._is_excluded(full_url):
                continue
            (priority if self._is_priority(full_url) else normal).append(full_url)
        return priority, normal

    def _decode_bytes(self, raw: bytes) -> str:
        """把网页字节解码为文本。

        **不要**用「哪种编码解出的汉字更多」来打分择优：GBK 是双字节编码，
        几乎任意字节对都能映射成某个汉字，于是把一份 UTF-8 网页按 GBK 解码
        往往能得到**更多**汉字（只是全是乱码），打分法会稳定地选错，
        产出「婀樻江澶у鍑虹増绀」这种乱码文本。

        可靠顺序是：
        1. **UTF-8 严格解码**——能通过校验基本就是 UTF-8；反之 GBK 中文网页
           极少能通过 UTF-8 严格校验，因此这是一条高置信度的判据；
        2. 页面自己声明的 charset（`<meta charset>` / `Content-Type`）；
        3. 回退 GB18030（GBK 的超集）。

        注意**不能**反过来把「声明」放在第一位：不少老站点在 `<meta>` 里
        声明 `gb2312`，实际却输出 UTF-8，按声明解码同样会得到乱码。
        """
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass

        declared = self._declared_charset(raw)
        if declared:
            encoding = {"gb2312": "gb18030", "gbk": "gb18030", "utf8": "utf-8"}.get(
                declared, declared
            )
            try:
                return raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                pass

        return raw.decode("gb18030", errors="replace")

    @staticmethod
    def _declared_charset(raw: bytes) -> str:
        """从 <meta charset> / <meta http-equiv> 中提取声明的编码。"""
        head = raw[:4096].lower()
        match = re.search(rb"charset\s*=\s*[\"']?\s*([a-z0-9_\-]+)", head)
        if not match:
            return ""
        return match.group(1).decode("ascii", "ignore").strip()

    def _html_to_text(self, html: str) -> str:
        """HTML → 正文纯文本。

        关键：**把导航菜单/页脚从正文中剥离**。此前用正则粗暴去标签，
        会把每个页面顶部那段一模一样的大导航菜单当成正文，导致
        「首页 学校概况 院系设置 …」这类无信息量的样板文字污染语料，
        检索时频繁命中导航块而非真正的内容页。

        策略按精度从高到低：
        1. 优先取正文容器（博达 CMS 的 #vsb_content / .v_news_content 等）；
        2. 回退到 <body>，先删掉导航/页脚容器，再取剩余文本。
        """
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_DROP_ALWAYS):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 先找正文容器：必须在删除 nav/header 等标签**之前**做，
        # 否则一旦正文被这些容器的父级包裹（如 <section><div class=header>…），
        # 就会连同正文一起被删掉。
        body_text = ""
        for selector in CONTENT_SELECTORS:
            node = soup.select_one(selector)
            if node:
                candidate = node.get_text("\n", strip=True)
                if len(re.sub(r"\s", "", candidate)) >= 100:
                    body_text = candidate
                    break

        if not body_text:
            for tag in soup(_DROP_TAGS):
                tag.decompose()
            body = soup.body or soup
            for el in list(body.find_all(True)):
                if getattr(el, "decomposed", False):
                    continue
                tokens: set[str] = set()
                for attr in ("class", "id"):
                    value = el.get(attr)
                    if not value:
                        continue
                    if isinstance(value, str):
                        value = [value]
                    for item in value:
                        tokens.update(re.split(r"[^a-zA-Z0-9]+", item.lower()))
                if tokens & _JUNK_TOKENS:
                    el.decompose()
            body_text = body.get_text("\n", strip=True)

        if title and title not in body_text:
            body_text = f"{title}\n{body_text}"
        return body_text

    def _text_from_bytes(self, raw: bytes) -> str:
        html = self._decode_bytes(raw)
        if len(html) < 50:
            return ""
        text = self._html_to_text(html)
        text = re.sub(r"[ \t\u3000]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def crawl(
        self, start_urls: list[str], output_dir: Optional[str] = None
    ) -> tuple[list[str], list[str]]:
        """广度优先爬取，支持断点续爬。

        使用**单一全局队列**（而非每个起始站点一条独立队列），原因有二：
        1. 恢复进度时必须能还原"待爬边界"，队列本身就是那份状态；
        2. 全局页数预算才有意义——否则某个大站点可以无限占用时间。

        每处理 STATE_SAVE_EVERY 个页面把 `visited + pending` 落盘，
        中断后重新运行会从断点继续，而不是从头再爬一遍。
        """
        file_links: list[str] = []
        page_links: list[str] = []
        site_counts: dict[str, int] = {}

        queue: deque[tuple[str, int]] = deque(self._pending)
        self._queued = {url for url, _ in queue}
        if not queue:
            for start_url in start_urls:
                url = start_url.rstrip("/")
                queue.append((url, 0))
                self._queued.add(url)
            logger.info("起始站点 %d 个，入队", len(start_urls))
        else:
            logger.info("从断点恢复，队列中已有 %d 个待爬 URL", len(queue))

        while queue:
            if self._pages_this_run >= self.max_total_pages:
                logger.info(
                    "已达本次全局页数预算 %d，剩余 %d 个 URL 留待下次续爬",
                    self.max_total_pages,
                    len(queue),
                )
                break

            url, depth = queue.popleft()

            if url in self._visited:
                continue
            if depth > self.max_depth:
                continue
            if not self._is_allowed_domain(url):
                continue
            if self._is_excluded(url):
                continue

            host = urlparse(url).hostname or ""
            count = site_counts.get(host, 0)
            if count >= self.max_pages_per_site:
                continue

            self._visited.add(url)
            site_counts[host] = count + 1
            self._pages_this_run += 1

            if self._pages_this_run % STATE_SAVE_EVERY == 0:
                self._pending = list(queue)
                self.save_state()
                logger.info(
                    "进度：本次已处理 %d 页 / 累计 %d 页 / 待爬 %d 个 / 落盘文本 %d 篇",
                    self._pages_this_run,
                    len(self._visited),
                    len(queue),
                    len(self._content_saved),
                )

            if self._is_file_link(url):
                file_links.append(url)
                logger.info("found file: %s", url)
                continue

            try:
                # 先查本地 HTML 缓存：命中则完全跳过网络请求。
                # 这让「加深层级重爬」「改进抽取规则重跑」都变成接近零成本的操作。
                cached = self._load_cached_html(url)
                if cached is not None:
                    raw = cached
                    self._cache_hits += 1
                else:
                    raw = self._fetch_raw(url)
                    if raw is None:
                        continue
                    # 缓存原始 HTML，便于日后改进抽取规则时离线重抽
                    self._save_html(raw, url)

                soup = BeautifulSoup(self._decode_bytes(raw), "lxml")
                page_links.append(url)
                marker = "*" if self._is_content_page(url) else " "
                logger.info("%s page: %s (depth=%d)%s", marker, url, depth,
                            " [cache]" if cached is not None else "")

                if output_dir and url not in self._content_saved:
                    text = self._text_from_bytes(raw)
                    if text and len(text) > 200:
                        self._save_text(text, url, output_dir)
                        self._content_saved.add(url)
                        logger.info("saved: %s (%d chars)", url, len(text))

                if depth < self.max_depth:
                    priority_links, normal_links = self._extract_links(soup, url)
                    # 优先栏目插到队首，避免被 max_pages_per_site 提前截断
                    priority_items, normal_items = [], []
                    for link in priority_links:
                        if self._enqueueable(link, site_counts):
                            priority_items.append((link, depth + 1))
                            self._queued.add(link)
                    for link in normal_links:
                        if self._enqueueable(link, site_counts):
                            normal_items.append((link, depth + 1))
                            self._queued.add(link)
                    queue.extend(normal_items)
                    if priority_items:
                        queue.extendleft(reversed(priority_items))

            except requests.RequestException as e:
                logger.warning("request failed: %s", e)
                continue
            except Exception as e:
                logger.warning("unexpected error at %s: %s", url, e)
                continue

            # 礼貌延时只在真正发了网络请求时才需要
            if cached is None:
                time.sleep(self.delay)

        self._pending = list(queue)
        self.save_state()

        if self._cache_hits:
            logger.info("本次复用本地 HTML 缓存 %d 页（未发起网络请求）", self._cache_hits)

        return page_links, list(set(file_links))

    def _fetch_raw(self, url: str) -> Optional[bytes]:
        """抓取页面 HTML；开启渲染时，对拿不到正文的页面改用浏览器。

        返回 None 表示这个 URL 不值得收录（非 200、非 HTML、渲染也失败）。
        """
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as e:
            logger.warning("request failed: %s", e)
            return None

        is_html = "text/html" in resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and is_html:
            # 正文过短 → 多半是 JS 骨架页，正文由前端异步加载
            if not self.render or len(self._text_from_bytes(resp.content)) >= 200:
                return resp.content
            logger.info("疑似 JS 渲染（正文过短），改用浏览器：%s", url)

        if not self.render:
            return None

        if not (resp.status_code == 200 and is_html):
            logger.info(
                "普通请求被拒（%s %s），改用浏览器：%s",
                resp.status_code,
                resp.headers.get("Content-Type", "-"),
                url,
            )
        return self._render(url)

    def _render(self, url: str) -> Optional[bytes]:
        if self._renderer is None:
            from .renderer import Renderer

            self._renderer = Renderer()
        html = self._renderer.fetch(url)
        if html is None:
            return None
        # 渲染后仍无实质正文的（如全站纯导航），直接丢弃
        if len(self._text_from_bytes(html)) < 200:
            return None
        return html

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _load_cached_html(self, url: str) -> Optional[bytes]:
        """读取本地缓存的原始 HTML；没有缓存返回 None。"""
        if not self._html_cache_dir:
            return None
        filepath = self._html_cache_dir / f"{self._safe_name(url)}.html"
        try:
            if filepath.exists() and filepath.stat().st_size > 0:
                return filepath.read_bytes()
        except OSError:
            return None
        return None

    @staticmethod
    def _safe_name(url: str) -> str:
        """URL → 安全的文件名主干（与 data/texts 下的命名保持一致）。"""
        parsed = urlparse(url)
        safe_name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\-_]", "_", parsed.netloc + parsed.path)
        return safe_name[:120] if len(safe_name) > 120 else safe_name

    def _save_text(self, text: str, url: str, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self._safe_name(url)}.txt"
        if not filepath.exists():
            filepath.write_text(text, encoding="utf-8")

    def _save_html(self, raw: bytes, url: str) -> None:
        """缓存原始 HTML（同名 .html），用于日后离线重抽文本。"""
        if not self._html_cache_dir:
            return
        self._html_cache_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._html_cache_dir / f"{self._safe_name(url)}.html"
        if not filepath.exists():
            filepath.write_bytes(raw)

    @property
    def visited_count(self) -> int:
        return len(self._visited)
