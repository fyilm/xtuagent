"""检索辅助单元测试。"""

from xtuagent.rag.retriever import build_context
from xtuagent.rag.vector_store import RetrievedDoc


def test_build_context_includes_numbered_sources():
    docs = [
        RetrievedDoc(content="内容A", source="a.txt", score=0.91),
        RetrievedDoc(content="内容B", source="b.txt", score=0.82),
    ]
    context = build_context(docs)

    assert "[1] 来源：a.txt" in context
    assert "[2] 来源：b.txt" in context
    assert "内容A" in context
    assert "内容B" in context
    assert "---" in context


def test_build_context_omits_raw_score():
    """相似度不写进上下文：分数未标定时会误导模型判断相关性。"""
    context = build_context([RetrievedDoc(content="内容", source="a.txt", score=0.876)])

    assert "0.876" not in context
    assert "0.88" not in context
    assert "相关度" not in context


def test_build_context_empty():
    assert build_context([]) == ""
