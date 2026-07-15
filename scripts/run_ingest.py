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
from xtuagent.loader import load_documents
from xtuagent.splitter import split_documents
from xtuagent.vector_store import build_vector_store, load_vector_store
from xtuagent.retriever import search_similar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_ingest")


def main():
    parser = argparse.ArgumentParser(description="文档入库：向量化并存入ChromaDB")
    parser.add_argument("--docs-dir", default=config.texts_dir, help="文本文件目录")
    parser.add_argument("--persist-dir", default=config.chroma_persist_dir, help="向量库持久化目录")
    parser.add_argument("--chunk-size", type=int, default=config.chunk_size, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=config.chunk_overlap, help="文本块重叠")
    parser.add_argument("--test", type=str, default=None, help="入库后执行测试查询")
    args = parser.parse_args()

    config.chunk_size = args.chunk_size
    config.chunk_overlap = args.chunk_overlap

    logger.info("===== loading documents =====")
    documents = load_documents(args.docs_dir)
    if not documents:
        logger.error("no documents found in %s", args.docs_dir)
        sys.exit(1)
    logger.info("loaded %d documents", len(documents))

    logger.info("===== splitting chunks =====")
    chunks = split_documents(documents)
    logger.info("produced %d chunks", len(chunks))

    logger.info("===== building vector store =====")
    vs = build_vector_store(chunks, persist_dir=args.persist_dir)
    logger.info("vector store ready at %s", args.persist_dir)

    if args.test:
        logger.info("===== test query: %s =====", args.test)
        docs = search_similar(vs, args.test)
        for i, doc in enumerate(docs, 1):
            logger.info("result %d: [%s] %s...", i, doc.metadata.get("source_file"), doc.page_content[:100])


if __name__ == "__main__":
    main()
