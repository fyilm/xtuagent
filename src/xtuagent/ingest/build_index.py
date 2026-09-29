"""索引构建入口：加载 → 分块 → 嵌入 → 持久化。"""

import logging
import time
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..rag.vector_store import build_index as build_faiss_index
from .loader import load_documents
from .splitter import split_documents

logger = logging.getLogger(__name__)


def run_build(
    docs_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """执行完整索引构建流程，返回统计信息。

    分块参数通过参数传递而非改写全局 `settings`：
    避免一次调用永久污染后续同进程调用的配置（例如对比实验脚本）。
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    start = time.time()

    logger.info("===== 1/3 加载文档 =====")
    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"未找到任何文档：{docs_dir or settings.texts_dir}")

    logger.info("===== 2/3 语义分块 =====")
    chunks = split_documents(documents, chunk_size=size, chunk_overlap=overlap)
    if not chunks:
        raise RuntimeError("分块结果为空，请检查文档内容")

    logger.info("===== 3/3 嵌入并构建索引 =====")
    service = build_faiss_index(
        chunks,
        index_dir=index_dir,
        documents=len(documents),
        chunk_size=size,
        chunk_overlap=overlap,
    )

    elapsed = time.time() - start
    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "vectors": service.count,
        "elapsed_s": round(elapsed, 1),
        "index_dir": str(index_dir or settings.index_dir),
    }
    logger.info("索引构建完成：%s", stats)
    return stats
