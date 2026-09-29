#!/usr/bin/env python
"""从缓存的原始 HTML 重新抽取正文 → data/texts/（**无需联网**）。

用途：正文抽取规则（正文容器选择、编码判定、导航剥离）会不断调整，
每改一次就整站重爬一次是不可接受的。`crawl.py` 会把每个页面的原始 HTML
缓存到 `data/raw_html/`，本脚本直接对缓存重跑抽取，秒级完成。

用法::

    uv run python scripts/extract_texts.py            # 重抽到 data/texts
    uv run python scripts/extract_texts.py --dry-run  # 只统计，不写文件
"""

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.crawler.spider import Spider

logger = logging.getLogger("extract_texts")

MIN_CHARS = 200


def main() -> None:
    parser = argparse.ArgumentParser(description="从缓存 HTML 重新抽取正文")
    parser.add_argument("--html-dir", type=Path, default=_ROOT / "data" / "raw_html")
    parser.add_argument("--out-dir", type=Path, default=settings.texts_dir)
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    setup_logging()

    if not args.html_dir.exists():
        logger.error(
            "HTML 缓存目录不存在：%s\n"
            "请先运行 crawl.py（会自动缓存原始 HTML），或改用 --html-dir 指定目录。",
            args.html_dir,
        )
        return

    spider = Spider(prefer_ipv4=False)
    html_files = sorted(args.html_dir.glob("*.html"))
    logger.info("发现 %d 个缓存页面", len(html_files))

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped_small = 0
    for html_path in html_files:
        raw = html_path.read_bytes()
        text = spider._text_from_bytes(raw)
        if not text or len(text) <= MIN_CHARS:
            skipped_small += 1
            continue
        if not args.dry_run:
            (args.out_dir / f"{html_path.stem}.txt").write_text(text, encoding="utf-8")
        written += 1

    verb = "将写出" if args.dry_run else "已写出"
    logger.info(
        "完成：%s %d 篇文本；丢弃 %d 篇（正文过短，多为纯导航页）",
        verb,
        written,
        skipped_small,
    )
    if not args.dry_run:
        total = len(list(args.out_dir.glob("*.txt")))
        logger.info("文本目录 %s 现有 %d 篇", args.out_dir, total)


if __name__ == "__main__":
    main()
