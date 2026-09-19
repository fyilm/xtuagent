#!/usr/bin/env python
"""爬取官网文档：爬取 → 下载 → 提取文本。"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.crawler.downloader import Downloader
from xtuagent.crawler.site_config import TARGET_SITES
from xtuagent.crawler.spider import Spider
from xtuagent.crawler.text_extractor import TextExtractor

logger = logging.getLogger("crawl")


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取湘潭大学官网文档")
    parser.add_argument("--max-depth", type=int, default=settings.crawl_max_depth)
    parser.add_argument("--delay", type=float, default=settings.crawl_delay)
    parser.add_argument("--timeout", type=int, default=settings.crawl_timeout)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    setup_logging()
    settings.ensure_dirs()

    start_urls = [site["url"] for site in TARGET_SITES]
    logger.info("目标站点数：%d", len(start_urls))

    logger.info("===== 1/3 爬取页面 =====")
    spider = Spider(max_depth=args.max_depth, delay=args.delay, timeout=args.timeout)
    pages, files = spider.crawl(start_urls, output_dir=str(settings.texts_dir))
    logger.info("爬取页面 %d 个，发现文件链接 %d 个", len(pages), len(files))

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

    logger.info("爬取完成，文本目录：%s", settings.texts_dir)


if __name__ == "__main__":
    main()
