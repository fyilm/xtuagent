"""FAISS 向量库服务：构建、加载、检索、元数据校验。"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..core.config import settings
from ..core.exceptions import IndexMetaMismatchError, IndexNotReadyError
from .embeddings import LangchainEmbeddingAdapter

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "index_meta.json"


@dataclass
class RetrievedDoc:
    """检索结果。score 为 (0,1] 的相似度，越大越相关。"""

    content: str
    source: str
    score: float


@dataclass
class IndexMeta:
    created_at: str = ""
    embedding_model: str = ""
    dimension: int = 0
    vector_count: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    documents: int = 0
    app_version: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "IndexMeta":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class VectorStoreService:
    """FAISS 索引的统一访问入口。"""

    def __init__(self, index: FAISS, meta: Optional[IndexMeta] = None) -> None:
        self._index = index
        self.meta = meta

    @property
    def count(self) -> int:
        return int(self._index.index.ntotal)

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        k = top_k or settings.retriever_top_k
        pairs = self._index.similarity_search_with_score(query, k=k)
        results: List[RetrievedDoc] = []
        for doc, distance in pairs:
            score = 1.0 / (1.0 + float(distance))
            source = (
                doc.metadata.get("source_file")
                or doc.metadata.get("source")
                or "unknown"
            )
            results.append(
                RetrievedDoc(
                    content=doc.page_content,
                    source=str(source),
                    score=round(score, 4),
                )
            )
        return results

    def health(self) -> dict:
        return {
            "count": self.count,
            "dimension": int(self._index.index.d),
            "meta": self.meta.to_dict() if self.meta else None,
        }


def build_index(
    chunks: List[Document],
    index_dir: Optional[Path] = None,
    documents: int = 0,
    embedding_fn=None,
) -> VectorStoreService:
    """构建并持久化 FAISS 索引，同时写入元数据文件。"""
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedding_fn = embedding_fn or LangchainEmbeddingAdapter()

    logger.info("构建 FAISS 索引：%d 个文本块", len(chunks))
    index = FAISS.from_documents(documents=chunks, embedding=embedding_fn)
    index.save_local(str(index_dir))

    meta = IndexMeta(
        created_at=datetime.now().isoformat(timespec="seconds"),
        embedding_model=settings.embedding_model,
        dimension=int(index.index.d),
        vector_count=int(index.index.ntotal),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        documents=documents,
        app_version=settings.app_version,
    )
    (index_dir / META_FILE).write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("索引已保存：%s（%d 条向量）", index_dir, meta.vector_count)
    return VectorStoreService(index, meta)


def load_index(
    index_dir: Optional[Path] = None,
    embedding_fn=None,
) -> VectorStoreService:
    """加载 FAISS 索引，并校验元数据。"""
    index_dir = Path(index_dir or settings.index_dir)
    faiss_file = index_dir / INDEX_FILE
    if not faiss_file.exists():
        raise IndexNotReadyError(f"索引不存在：{faiss_file}，请先运行构建索引")

    meta: Optional[IndexMeta] = None
    meta_file = index_dir / META_FILE
    if meta_file.exists():
        meta = IndexMeta.from_dict(json.loads(meta_file.read_text(encoding="utf-8")))
        if meta.embedding_model and meta.embedding_model != settings.embedding_model:
            raise IndexMetaMismatchError(
                f"索引由「{meta.embedding_model}」构建，当前配置为「{settings.embedding_model}」，请重建索引"
            )

    embedding_fn = embedding_fn or LangchainEmbeddingAdapter()
    index = FAISS.load_local(
        str(index_dir), embedding_fn, allow_dangerous_deserialization=True
    )
    service = VectorStoreService(index, meta)
    logger.info("索引加载完成：%d 条向量（%s）", service.count, index_dir)
    return service
