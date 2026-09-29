"""RAG 流水线单元测试：阈值兜底、top_k 透传、流式事件序列。

使用桩向量库 + 假聊天模型，不联网、不加载嵌入模型。
"""

from typing import List

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from xtuagent.rag.pipeline import FALLBACK_ANSWER, RAGPipeline
from xtuagent.rag.vector_store import RetrievedDoc


class StubVectorStore:
    """可编程桩：记录调用参数，返回预设检索结果。"""

    def __init__(self, docs: List[RetrievedDoc] | None = None) -> None:
        self.docs = docs or []
        self.calls: List[dict] = []

    def search(self, query, top_k=None, min_score=None, max_per_source=None):
        self.calls.append({"query": query, "top_k": top_k})
        return list(self.docs)


def make_llm(text: str = "根据《湘潭大学学分制管理规定》，平均学分绩点低于 1.5 会收到学业警告 [1]。"):
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


SAMPLE_DOC = RetrievedDoc(
    content="平均学分绩点低于 1.5 的学生将收到学业警告。",
    source="学分制管理规定.txt",
    score=0.83,
)


def test_ask_returns_fallback_when_retrieval_is_empty():
    """检索被阈值过滤为空时，必须返回兜底回答而不是让模型硬编。"""
    store = StubVectorStore(docs=[])
    pipeline = RAGPipeline(store, llm=make_llm())

    answer = pipeline.ask("量子力学怎么选课？")

    assert answer.text == FALLBACK_ANSWER
    assert answer.sources == []
    assert answer.elapsed_ms >= 0


def test_ask_forwards_top_k_override():
    """服务端传入的 top_k 必须真正生效（此前该参数被静默丢弃）。"""
    store = StubVectorStore(docs=[SAMPLE_DOC])
    pipeline = RAGPipeline(store, llm=make_llm(), top_k=5)

    pipeline.ask("学分绩点要求", top_k=2)

    assert store.calls[0]["top_k"] == 2


def test_ask_uses_pipeline_default_top_k_when_not_overridden():
    store = StubVectorStore(docs=[SAMPLE_DOC])
    pipeline = RAGPipeline(store, llm=make_llm(), top_k=4)

    pipeline.ask("学分绩点要求")

    assert store.calls[0]["top_k"] == 4


def test_ask_returns_generated_text_and_sources():
    store = StubVectorStore(docs=[SAMPLE_DOC])
    pipeline = RAGPipeline(store, llm=make_llm("答案内容"))

    answer = pipeline.ask("学分绩点要求")

    assert answer.text == "答案内容"
    assert answer.sources == [SAMPLE_DOC]


def test_ask_stream_emits_sources_then_delta_then_done():
    store = StubVectorStore(docs=[SAMPLE_DOC])
    pipeline = RAGPipeline(store, llm=make_llm("流式答案"))

    events = list(pipeline.ask_stream("学分绩点要求"))

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["file"] == "学分制管理规定.txt"
    assert events[-1]["type"] == "done"
    assert "elapsed_ms" in events[-1]

    deltas = [e for e in events if e["type"] == "delta"]
    assert deltas, "应至少产出一个 delta 事件"
    assert "流式答案" in "".join(e["text"] for e in deltas)


def test_ask_stream_short_circuits_on_empty_retrieval():
    store = StubVectorStore(docs=[])
    pipeline = RAGPipeline(store, llm=make_llm())

    events = list(pipeline.ask_stream("无关问题"))

    assert events[0] == {"type": "sources", "sources": []}
    assert any(e["type"] == "delta" and e["text"] == FALLBACK_ANSWER for e in events)
    assert events[-1]["type"] == "done"


def test_ask_retries_then_raises_llm_invocation_error(monkeypatch):
    """重试策略：全部失败后抛领域异常，而不是泄漏底层异常。"""
    from langchain_core.runnables import RunnableLambda

    from xtuagent.core.config import settings
    from xtuagent.core.exceptions import LLMInvocationError

    attempts = {"n": 0}

    def boom(_value):
        attempts["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr("xtuagent.rag.pipeline.time.sleep", lambda _s: None)

    store = StubVectorStore(docs=[SAMPLE_DOC])
    pipeline = RAGPipeline(store, llm=RunnableLambda(boom))

    with pytest.raises(LLMInvocationError):
        pipeline.ask("学分绩点要求")
    assert attempts["n"] == 2, "应按 llm_max_retries 重试"
