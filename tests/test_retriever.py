"""检索辅助单元测试。"""

from xtuagent.rag.retriever import build_context
from xtuagent.rag.vector_store import RetrievedDoc


def test_build_context_includes_source_labels():
    docs = [
        RetrievedDoc(content="内容A", source="a.txt", score=0.9),
        RetrievedDoc(content="内容B", source="b.txt", score=0.8),
    ]
    context = build_context(docs)

    assert "[来源: a.txt]" in context
    assert "[来源: b.txt]" in context
    assert "内容A" in context
    assert "内容B" in context
    assert "---" in context


def test_build_context_empty():
    assert build_context([]) == ""
