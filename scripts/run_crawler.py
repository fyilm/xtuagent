#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.config import config
from xtuagent.crawler.spider import Spider
from xtuagent.crawler.downloader import Downloader
from xtuagent.crawler.text_extractor import TextExtractor
from xtuagent.crawler.site_config import TARGET_SITES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_crawler")


def main():
    parser = argparse.ArgumentParser(description="爬取湘潭大学官网文档")
    parser.add_argument("--max-depth", type=int, default=config.crawl_max_depth, help="最大爬取深度")
    parser.add_argument("--delay", type=float, default=config.crawl_delay, help="请求间隔(秒)")
    parser.add_argument("--timeout", type=int, default=config.crawl_timeout, help="请求超时(秒)")
    parser.add_argument("--workers", type=int, default=5, help="下载并发数")
    parser.add_argument("--pdf-dir", default=config.raw_pdfs_dir, help="PDF输出目录")
    parser.add_argument("--text-dir", default=config.texts_dir, help="文本输出目录")
    args = parser.parse_args()

    start_urls = [s["url"] for s in TARGET_SITES]
    logger.info("target sites: %d", len(start_urls))

    logger.info("===== phase 1: crawling =====")
    spider = Spider(max_depth=args.max_depth, delay=args.delay, timeout=args.timeout)
    page_links, file_links = spider.crawl(start_urls)
    logger.info("crawled %d pages, found %d file links", len(page_links), len(file_links))

    if not file_links:
        logger.warning("no files found, exiting")
        return

    logger.info("===== phase 2: downloading =====")
    downloader = Downloader(output_dir=args.pdf_dir, timeout=args.timeout, workers=args.workers)
    downloaded = downloader.download_batch(file_links)
    logger.info("downloaded %d/%d files", len(downloaded), len(file_links))

    if not downloaded:
        logger.warning("no files downloaded, exiting")
        return

    logger.info("===== phase 3: text extraction =====")
    extractor = TextExtractor(output_dir=args.text_dir)
    extracted = extractor.extract_batch(downloaded)
    logger.info("extracted %d/%d files to text", len(extracted), len(downloaded))

    logger.info("===== done =====")
    logger.info("ready to run: uv run scripts/run_ingest.py")


if __name__ == "__main__":
    main()
