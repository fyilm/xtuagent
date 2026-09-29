#!/usr/bin/env python
r"""把一批外部 URL 或本地文件灌入语料库（供官网之外的渠道使用）。

适用于微信公众号文章、知乎/贴吧等站外内容，以及手头已有的
PDF/Word/Markdown 文件——官方爬虫（crawl.py）只认 xtu.edu.cn 域，
这些「其他信息渠道」走本脚本。

用法::

    # 从文件读取 URL（每行一个，# 开头为注释）
    uv run python scripts/ingest_urls.py urls.txt --source wechat

    # 直接给 URL
    uv run python scripts/ingest_urls.py https://mp.weixin.qq.com/s/xxxx --source wechat

    # 灌入本地文件（pdf/docx/md/txt/html）
    uv run python scripts/ingest_urls.py ./docs/*.pdf --source handbook

    # 浏览器里「另存为」的页面（抓不到的站点用这招兜底）
    uv run python scripts/ingest_urls.py ./saved_pages/*.html --source portal

    # 需要登录的页面：先用浏览器导出 Cookie 存成本地文件
    uv run python scripts/ingest_urls.py urls.txt --cookie-file .\my.cookies.txt

    # 只看会写什么，不落盘
    uv run python scripts/ingest_urls.py urls.txt --source wechat --dry-run

灌完后照常运行 `uv run python scripts/build_index.py` 重建索引即可。
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import requests

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.crawler.cookies import load_cookies
from xtuagent.crawler.downloader import Downloader
from xtuagent.crawler.spider import Spider
from xtuagent.crawler.text_extractor import TextExtractor

logger = logging.getLogger("ingest_urls")

MIN_CHARS = 100
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
_ATTACHMENT_SUFFIXES = {".pdf", ".doc", ".docx", ".md", ".txt"}


def _text_from_file(path: Path, extractor: TextExtractor, spider: Spider = None) -> str:
    """按扩展名分派到对应的解析器。

    `.html/.htm` 走网页抽取逻辑（剥导航、取正文容器）——这条路径用于
    **用户在浏览器里直接「另存为」的页面**：对于抓不到的站点
    （WAF 403 / JS 渲染 / 需登录），人工保存后灌入是最稳的兜底。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extractor.extract_pdf(str(path)) or ""
    if suffix == ".docx":
        return extractor.extract_docx(str(path)) or ""
    if suffix in (".html", ".htm"):
        spider = spider or Spider(prefer_ipv4=False)
        return spider._text_from_bytes(path.read_bytes())
    if suffix in (".md", ".txt"):
        return extractor._clean_text(path.read_text(encoding="utf-8", errors="replace"))
    return ""


def _read_targets(items: List[str]) -> List[str]:
    """把「URL 列表文件」和「直接给出的 URL」统一展开为一个列表。"""
    targets: List[str] = []
    for item in items:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    targets.append(line)
        else:
            targets.append(item)
    return targets


def _content_key(text: str) -> str:
    return hashlib.blake2b("".join(text.split()).encode("utf-8"), digest_size=16).hexdigest()


def _existing_keys(texts_dir: Path) -> set:
    keys = set()
    for f in texts_dir.glob("*.txt"):
        try:
            keys.add(_content_key(f.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="把外部 URL / 本地文件灌入语料库")
    parser.add_argument("targets", nargs="+", help="URL、URL 列表文件，或本地文件路径")
    parser.add_argument("--source", default="external", help="来源前缀，用于文件命名与溯源")
    parser.add_argument("--out-dir", type=Path, default=settings.texts_dir)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--cookie-file",
        default=None,
        help="需要登录的页面：从浏览器导出的 Cookie 文件（Netscape cookies.txt 或 JSON）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    setup_logging()
    targets = _read_targets(args.targets)
    logger.info("待处理 %d 个目标（来源标记：%s）", len(targets), args.source)

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    known = _existing_keys(args.out_dir)

    cookies = load_cookies(args.cookie_file)
    spider = Spider(prefer_ipv4=False, cookies=cookies)
    extractor = TextExtractor(output_dir=str(args.out_dir))
    downloader = Downloader(output_dir=str(settings.raw_pdfs_dir), timeout=args.timeout)
    session = requests.Session()
    session.headers.update(_HEADERS)
    if cookies:
        session.cookies.update(cookies)

    added = skipped_dup = failed = 0

    for target in targets:
        try:
            local = Path(target)
            if local.is_file():
                text, origin = _text_from_local(local, extractor, spider)
            else:
                text, origin = _text_from_url(target, session, spider, extractor, downloader, args.timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("处理失败 %s：%s", target, exc)
            failed += 1
            continue

        if not text or len(text) < MIN_CHARS:
            logger.warning("正文过短或为空，跳过：%s", target)
            failed += 1
            continue

        key = _content_key(text)
        if key in known:
            logger.info("内容重复，跳过：%s", target)
            skipped_dup += 1
            continue
        known.add(key)

        if not args.dry_run:
            stem = _safe_stem(origin)
            (args.out_dir / f"{args.source}_{stem}.txt").write_text(text, encoding="utf-8")
        added += 1
        logger.info("已收录（%d 字）：%s", len(text), target)

    verb = "将收录" if args.dry_run else "已收录"
    logger.info(
        "完成：%s %d 条；跳过重复 %d 条；失败 %d 条",
        verb, added, skipped_dup, failed,
    )
    if not args.dry_run:
        logger.info("请运行 `uv run python scripts/build_index.py` 重建索引")


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = "".join(c if c.isalnum() or c in "-_一" else "_" for c in stem)
    return cleaned[:80] or hashlib.blake2b(name.encode("utf-8"), digest_size=6).hexdigest()


def _text_from_local(path: Path, extractor: TextExtractor, spider: Spider):
    if path.suffix.lower() == ".doc":
        logger.warning("老版 .doc 无法直接解析，请另存为 .docx 或 .pdf：%s", path.name)
        return "", path.name
    return _text_from_file(path, extractor, spider), path.name


def _text_from_url(url, session, spider, extractor, downloader, timeout):
    """抓取一个外部 URL，返回 (正文, 用于命名的标识)。"""
    lower = url.lower().split("?")[0]
    if any(lower.endswith(s) for s in _ATTACHMENT_SUFFIXES) or spider._is_file_link(url):
        saved = downloader.download_one(url)
        if not saved:
            return "", url
        path = Path(saved)
        return _text_from_file(path, extractor), path.name

    resp = session.get(url, timeout=timeout, allow_redirects=True)
    if resp.status_code != 200:
        logger.warning("HTTP %d：%s", resp.status_code, url)
        return "", url
    content_type = resp.headers.get("Content-Type", "")
    looks_html = "html" in content_type.lower() or resp.content.lstrip()[:1] == b"<"
    if not looks_html:
        logger.warning("非 HTML 内容（%s）：%s", content_type, url)
        return "", url
    return spider._text_from_bytes(resp.content), url


if __name__ == "__main__":
    main()
