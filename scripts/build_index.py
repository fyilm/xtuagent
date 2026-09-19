#!/usr/bin/env python
"""构建向量索引：加载 → 分块 → 嵌入 → FAISS 持久化。"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging
from xtuagent.ingest.build_index import run_build


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 FAISS 向量索引")
    parser.add_argument("--docs-dir", type=Path, default=settings.texts_dir)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    stats = run_build(
        docs_dir=args.docs_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("\n===== 构建完成 =====")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
