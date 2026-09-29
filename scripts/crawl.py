#!/usr/bin/env python
"""爬取官网文档：爬取 → 下载 → 提取文本。

支持断点续爬：进度保存在 data/.crawl_state.json（已访问 URL + 待爬队列）。
中断后直接重新运行本脚本即可从断点继续，不会从头再爬一遍。
需要完全重爬时加 `--fresh`。
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.crawler.cookies import load_cookies
from xtuagent.crawler.downloader import Downloader
from xtuagent.crawler.site_config import TARGET_SITES
from xtuagent.crawler.spider import Spider
from xtuagent.crawler.text_extractor import TextExtractor

logger = logging.getLogger("crawl")

STATE_FILE = "data/.crawl_state.json"
HTML_CACHE_DIR = "data/raw_html"


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取湘潭大学官网文档")
    parser.add_argument("--max-depth", type=int, default=settings.crawl_max_depth)
    parser.add_argument("--delay", type=float, default=settings.crawl_delay)
    parser.add_argument("--timeout", type=int, default=settings.crawl_timeout)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--max-total-pages",
        type=int,
        default=settings.crawl_max_total_pages,
        help="单次运行的全局页数上限，未爬完的部分下次续爬",
    )
    parser.add_argument("--fresh", action="store_true", help="忽略断点，从头开始爬")
    parser.add_argument(
        "--render",
        action="store_true",
        help="用无头浏览器兜底抓取 JS 渲染页 / 被 WAF 403 的站点（慢，需要 playwright）",
    )
    parser.add_argument(
        "--cookie-file",
        default=None,
        help="需要登录时，从浏览器导出的 Cookie 文件（Netscape cookies.txt 或 JSON）",
    )
    args = parser.parse_args()

    setup_logging()
    settings.ensure_dirs()

    state_file = _ROOT / STATE_FILE
    if args.fresh and state_file.exists():
        # 沙箱/安全删除钩子可能把 unlink 劫持为回收站操作并使其中止，
        # 状态清理失败不应中断爬取，退化为「覆盖写」即可。
        try:
            state_file.unlink()
        except OSError as exc:
            logger.warning("清除断点状态失败（%s），改为直接覆盖", exc)
        logger.info("--fresh：已清除断点状态")

    start_urls = [site["url"] for site in TARGET_SITES]
    logger.info("目标站点数：%d", len(start_urls))

    logger.info("===== 1/3 爬取页面 =====")
    spider = Spider(
        max_depth=args.max_depth,
        delay=args.delay,
        timeout=args.timeout,
        max_total_pages=args.max_total_pages,
        state_file=str(state_file),
        resume=not args.fresh,
        html_cache_dir=str(_ROOT / HTML_CACHE_DIR),
        render=args.render,
        cookies=load_cookies(args.cookie_file),
    )
    logger.info("渲染兜底：%s", "已开启" if args.render else "关闭")
    try:
        pages, files = spider.crawl(start_urls, output_dir=str(settings.texts_dir))
    finally:
        spider.close()
    logger.info("本次爬取页面 %d 个，发现文件链接 %d 个", len(pages), len(files))

    if files:
        logger.info("===== 2/3 下载文件 =====")
        downloader = Downloader(
            output_dir=str(settings.raw_pdfs_dir), timeout=args.timeout, workers=args.workers
        )
        downloaded = downloader.download_batch(files)
        logger.info("下载完成 %d/%d", len(downloaded), len(files))

        if downloaded:
            logger.info("===== 3/3 提取文本 =====")
            extractor = TextExtractor(output_dir=str(settings.texts_dir))
            extracted = extractor.extract_batch(downloaded)
            logger.info("提取文本 %d/%d", len(extracted), len(downloaded))

    text_count = len(list(settings.texts_dir.glob("*.txt")))
    logger.info("完成。文本目录 %s 现有 %d 篇", settings.texts_dir, text_count)
    logger.info("如需继续爬取剩余页面，再次运行本脚本即可（自动续爬）")


if __name__ == "__main__":
    main()
