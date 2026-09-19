"""向量库单元测试（使用假嵌入，秒级完成，不加载真实模型）。"""

from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from xtuagent.core.exceptions import IndexNotReadyError
from xtuagent.rag.vector_store import (
    META_FILE,
    RetrievedDoc,
    build_index,
    load_index,
)


class FakeEmbeddings(Embeddings):
    """确定性假嵌入：基于字符 ASCII 值生成固定维度向量。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> List[float]:
        base = sum(ord(c) for c in text) % 97
        return [float(base), float((base * 7) % 89), float((base * 13) % 83), 1.0]


def _make_chunks(n: int = 6) -> List[Document]:
    return [
        Document(
            page_content=f"湘潭大学第{i}条规定内容，学分管理相关说明。",
            metadata={"source_file": f"doc_{i}.txt"},
        )
        for i in range(n)
    ]


def test_build_and_load_index(tmp_path):
    service = build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())

    assert service.count == 6
    assert service.meta is not None
    assert service.meta.vector_count == 6
    assert service.meta.documents == 6
    assert service.meta.dimension == 4
    assert (tmp_path / META_FILE).exists()

    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())
    assert loaded.count == 6
    assert loaded.meta is not None


def test_search_returns_scored_docs(tmp_path):
    build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    results = loaded.search("学分管理规定", top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievedDoc) for r in results)
    assert all(0 < r.score <= 1 for r in results)
    assert all(r.source.startswith("doc_") for r in results)


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(IndexNotReadyError):
        load_index(tmp_path, embedding_fn=FakeEmbeddings())


def test_load_meta_model_mismatch_raises(tmp_path):
    import json

    build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())
    meta_path = tmp_path / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["embedding_model"] = "different-model"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(Exception):
        load_index(tmp_path, embedding_fn=FakeEmbeddings())
