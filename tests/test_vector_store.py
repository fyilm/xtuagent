"""向量库单元测试（使用假嵌入，秒级完成，不加载真实模型）。"""

import json
import math
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from xtuagent.core.exceptions import IndexMetaMismatchError, IndexNotReadyError
from xtuagent.rag.vector_store import (
    META_FILE,
    RetrievedDoc,
    build_index,
    compute_corpus_fingerprint,
    load_index,
)


class FakeEmbeddings(Embeddings):
    """确定性假嵌入：字符哈希词袋。

    共享字符越多则向量越接近，因此可用余弦相似度模拟真实语义检索行为，
    同时完全离线、无需下载模型，适合单测。
    """

    dim = 512

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    @classmethod
    def _vec(cls, text: str) -> List[float]:
        vec = [0.0] * cls.dim
        for ch in text:
            vec[ord(ch) % cls.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


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
    assert service.meta.dimension == FakeEmbeddings.dim
    assert (tmp_path / META_FILE).exists()

    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())
    assert loaded.count == 6
    assert loaded.meta is not None


def test_search_returns_scored_docs(tmp_path):
    build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    # min_score=0 关闭阈值过滤，单独验证打分与排序行为
    # min_chunk_chars=0 / dedup_threshold=0：同时关掉碎片过滤与近似去重，
    # 这个用例只考察打分与排序，量化指标另有两个专门用例。
    results = loaded.search(
        "第1条规定内容 学分管理",
        top_k=3,
        min_score=0.0,
        min_chunk_chars=0,
        dedup_threshold=0.0,
    )

    assert len(results) == 3
    assert all(isinstance(r, RetrievedDoc) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert all(r.source.startswith("doc_") for r in results)
    assert results[0].score >= results[-1].score, "结果应按相关度降序返回"


def test_score_uses_cosine_semantics(tmp_path):
    """score 必须是真实余弦相似度：完全相同文本应接近 1.0。"""
    chunks = [Document(page_content="学分制管理规定", metadata={"source_file": "a.txt"})]
    build_index(chunks, index_dir=tmp_path, documents=1, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    results = loaded.search(
        "学分制管理规定", top_k=1, min_score=0.0, min_chunk_chars=0, dedup_threshold=0.0
    )
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_drops_tiny_chunks(tmp_path):
    """抽取残渣（「。」这类碎块）不得占用 top_k 名额。

    语料里真实存在 len<10 的块（`。`、`..`、`.7亿元`），它们没有信息量，
    却会和正经内容一起被送进模型。
    """
    chunks = [
        Document(page_content="。", metadata={"source_file": "junk.txt"}),
        Document(
            page_content="湘潭大学学分管理规定" * 5, metadata={"source_file": "real.txt"}
        ),
    ]
    build_index(chunks, index_dir=tmp_path, documents=2, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    both = loaded.search(
        "学分管理规定", top_k=2, min_score=0.0, min_chunk_chars=0, dedup_threshold=0.0
    )
    assert len(both) == 2

    kept = loaded.search(
        "学分管理规定", top_k=2, min_score=0.0, min_chunk_chars=40, dedup_threshold=0.0
    )
    assert [r.source for r in kept] == ["real.txt"]


def test_search_suppresses_near_duplicate_chunks(tmp_path):
    """跨文件同文不得把 top_k 占满。

    同一段样板文字会出现在几十个页面上，而 `max_per_source` 是按**来源文件**
    去重的，挡不住这种「不同文件、同一段文字」的情况。
    """
    # 注意填充文本必须是**非重复**的：若用「第一章总则」×20 这种周期串，
    # 5-gram 集合会退化到十几个元素，Jaccard 反而失真（≈0.85），
    # 那是测试用例的问题，不是去重逻辑的问题。
    filler = "".join(chr(0x4E00 + i) for i in range(220))
    base = "湘潭大学学分管理规定" + filler
    chunks = [
        Document(page_content=f"{base}{suffix}", metadata={"source_file": f"page_{i}.txt"})
        for i, suffix in enumerate("甲乙丙丁")
    ]
    build_index(chunks, index_dir=tmp_path, documents=4, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    raw = loaded.search(
        "学分管理规定", top_k=4, min_score=0.0, min_chunk_chars=0, dedup_threshold=0.0
    )
    assert sum(1 for r in raw if r.source.startswith("page_")) == 4

    deduped = loaded.search(
        "学分管理规定", top_k=4, min_score=0.0, min_chunk_chars=0, dedup_threshold=0.9
    )
    assert sum(1 for r in deduped if r.source.startswith("page_")) == 1


def test_threshold_filters_irrelevant_results(tmp_path):
    """低于阈值的结果必须被丢弃，使 pipeline 的兜底回答可达。"""
    build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    assert loaded.search("学分管理规定", top_k=3, min_score=0.95) == []


def test_max_per_source_limits_single_document(tmp_path):
    """同一来源的片段数受限，避免上下文被单个长文档占满。"""
    chunks = [
        Document(page_content=f"学分管理规定第{i}条", metadata={"source_file": "same.txt"})
        for i in range(5)
    ]
    build_index(chunks, index_dir=tmp_path, documents=1, embedding_fn=FakeEmbeddings())
    loaded = load_index(tmp_path, embedding_fn=FakeEmbeddings())

    results = loaded.search(
        "学分管理规定",
        top_k=5,
        min_score=0.0,
        max_per_source=2,
        min_chunk_chars=0,
        dedup_threshold=0.0,
    )
    assert len(results) == 2


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(IndexNotReadyError):
        load_index(tmp_path, embedding_fn=FakeEmbeddings())


def test_load_meta_model_mismatch_raises(tmp_path):
    build_index(_make_chunks(), index_dir=tmp_path, documents=6, embedding_fn=FakeEmbeddings())
    meta_path = tmp_path / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["embedding_model"] = "different-model"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(IndexMetaMismatchError):
        load_index(tmp_path, embedding_fn=FakeEmbeddings())


def test_fingerprint_detects_change_after_third_chunk():
    """回归测试：旧指纹只看前 3 条 + 总数，语料中间变更会导致旧嵌入被错误复用。"""
    original = ["甲", "乙", "丙", "丁", "戊"]
    mutated = ["甲", "乙", "丙", "改", "戊"]

    assert compute_corpus_fingerprint(original) != compute_corpus_fingerprint(mutated)


def test_fingerprint_is_stable_and_order_sensitive():
    assert compute_corpus_fingerprint(["a", "b"]) == compute_corpus_fingerprint(["a", "b"])
    assert compute_corpus_fingerprint(["a", "b"]) != compute_corpus_fingerprint(["b", "a"])
