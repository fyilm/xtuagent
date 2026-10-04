"""联网兜底流水线测试：拒答识别、触发时机、事件序列、失败降级。

全部使用桩模型与桩检索，不打真实网络（`tests/conftest.py` 默认关闭联网兜底，
本文件按需在单个用例内打开）。
"""

from typing import List

import pytest
from langchain_core.runnables import RunnableLambda, RunnableGenerator

from xtuagent.core.config import settings
from xtuagent.core.messages import fallback_answer, web_disclaimer, web_status
from xtuagent.rag import pipeline as pipeline_mod
from xtuagent.rag.pipeline import (
    FALLBACK_ANSWER,
    RAGPipeline,
    WEB_DISCLAIMER,
    is_unanswerable,
)
from xtuagent.rag.vector_store import RetrievedDoc
from xtuagent.rag.websearch import WebDoc

KB_ANSWER = "平均学分绩点低于 1.5 会收到学业警告 [1]。"
KB_REFUSAL = "根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员获取更详细的说明。"
WEB_ANSWER = "校内知识库未收录该内容，以下信息来自网络检索，请以学校官方通知为准。VPN 需先登录学校邮箱再下载客户端 [1]。"

SAMPLE_DOC = RetrievedDoc(
    content="平均学分绩点低于 1.5 的学生将收到学业警告。",
    source="学分制管理规定.txt",
    score=0.83,
)

WEB_DOC = WebDoc(
    title="校园网-湘潭大学",
    link="https://www.xtu.edu.cn/xysh/ggfw/xyw.htm",
    media="湘潭大学",
    content="校园网为全校师生提供免费 VPN 服务。",
)


class StubVectorStore:
    def __init__(self, docs: List[RetrievedDoc] | None = None) -> None:
        self.docs = docs if docs is not None else [SAMPLE_DOC]

    def search(self, query, top_k=None, min_score=None, max_per_source=None):
        return list(self.docs)


def make_llm(*responses: str):
    """按顺序回复预设文本的桩模型。

    `RunnableLambda` 既支持 invoke 也支持 stream（stream 时整段作为单个 chunk），
    因此同一条链可以同时覆盖同步与流式两条路径。
    """
    queue = list(responses)

    def _fn(_input):
        return queue.pop(0) if queue else "（桩模型已无预设回答）"

    return RunnableLambda(_fn)


def make_stream_llm(chunks):
    """按字符块流式输出的桩模型，用于验证扣留/补发行为。"""

    def _gen(_input):
        for chunk in chunks:
            yield chunk

    return RunnableGenerator(_gen)


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setattr(settings, "web_fallback_enabled", True)


@pytest.fixture
def stub_search(monkeypatch):
    """把 pipeline 命名空间里的 web_search 换成桩，记录调用次数。"""
    calls = []

    def _search(query, *args, **kwargs):
        calls.append(query)
        return [WEB_DOC]

    monkeypatch.setattr(pipeline_mod, "web_search", _search)
    return calls


# ---------------------------------------------------------------------------
# 拒答识别
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        KB_REFUSAL,
        "根据现有知识库暂未找到相关信息，建议咨询当地气象部门获取更准确的信息。",
        "知识库中未收录《高等数学》的课程安排，建议登录教务系统查询。",
        "很抱歉，知识库中没有关于该问题的信息。",
        "暂未找到相关依据，无法回答。",
        "知识库里不包含这类内容。",
    ],
)
def test_is_unanswerable_detects_refusals(text):
    assert is_unanswerable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        KB_ANSWER,
        # 这句里含「没有找到相关」，但它是正常回答的一部分，绝不能被误判；
        # 否则会把一个好答案换成网络答案，代价比漏判大得多。
        "学分绩点低于 1.5 会收到学业警告。如果没有找到相关豁免条款，请联系教务处。",
        "湘潭大学位于中国湖南省湘潭市 [2]。",
        "校园网提供 WEB、E-MAIL、DNS 等基本服务，并为全校师生提供免费 VPN 服务 [1]。",
        "",
    ],
)
def test_is_unanswerable_ignores_normal_answers(text):
    assert is_unanswerable(text) is False


# ---------------------------------------------------------------------------
# 同步问答
# ---------------------------------------------------------------------------


def test_ask_normal_answer_never_triggers_web_search(web_on, stub_search):
    """能答的问题绝不能产生搜索费用——这是成本底线。"""
    p = RAGPipeline(StubVectorStore(), llm=make_llm(KB_ANSWER))

    answer = p.ask("学分绩点要求")

    assert answer.text == KB_ANSWER
    assert answer.mode == "kb"
    assert answer.sources == [SAMPLE_DOC]
    assert answer.web_sources == []
    assert stub_search == [], "知识库答得出来时不该联网检索"


def test_ask_falls_back_to_web_when_kb_refuses(web_on, stub_search):
    p = RAGPipeline(StubVectorStore(), llm=make_llm(KB_REFUSAL, WEB_ANSWER))

    answer = p.ask("校园网 VPN 怎么用")

    assert answer.mode == "web"
    assert answer.text == WEB_DISCLAIMER + WEB_ANSWER
    assert answer.web_sources == [WEB_DOC]
    assert answer.sources == [], "联网作答时不该把不相关的知识库片段挂在来源里"
    assert stub_search == ["校园网 VPN 怎么用"]


def test_ask_refused_when_web_search_fails(web_on, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(pipeline_mod, "web_search", boom)
    p = RAGPipeline(StubVectorStore(), llm=make_llm(KB_REFUSAL))

    answer = p.ask("校园网 VPN 怎么用")

    assert answer.mode == "refused"
    assert answer.text == KB_REFUSAL, "联网失败必须退回原来的拒答话术"
    assert answer.web_sources == []


def test_ask_refused_when_web_returns_nothing(web_on, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "web_search", lambda *a, **k: [])
    p = RAGPipeline(StubVectorStore(), llm=make_llm(KB_REFUSAL))

    answer = p.ask("校园网 VPN 怎么用")

    assert answer.mode == "refused"
    assert answer.text == KB_REFUSAL


def test_ask_empty_retrieval_triggers_web(web_on, stub_search):
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm(WEB_ANSWER))

    answer = p.ask("今天长沙天气")

    assert answer.mode == "web"
    assert answer.text == WEB_DISCLAIMER + WEB_ANSWER
    assert answer.web_sources == [WEB_DOC]
    assert stub_search == ["今天长沙天气"]


def test_disclaimer_is_injected_by_server_not_left_to_model(web_on, stub_search):
    """免责声明必须由服务端强制加上。

    实测 glm-4-flash 会漏掉提示词里"先说明来源性质"这条要求，直接开讲。
    这里用一个**完全不提来源**的桩回答，验证声明依然存在——即不依赖模型自觉。
    """
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm("今天 26 度，晴。"))

    answer = p.ask("今天长沙天气")

    assert answer.text.startswith(WEB_DISCLAIMER)
    assert "非学校官方发布" in answer.text


def test_ask_web_disabled_keeps_old_behaviour(stub_search):
    """开关关闭时必须与改造前完全一致：不联网、直接拒答。"""
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm(KB_ANSWER))

    answer = p.ask("今天长沙天气")

    assert answer.mode == "refused"
    assert answer.text == FALLBACK_ANSWER
    assert stub_search == []


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------


def test_ask_stream_normal_answer_streams_incrementally(web_on, stub_search):
    chunks = ["平均学分绩点", "低于 1.5 ", "会收到学业警告 [1]。"]
    p = RAGPipeline(StubVectorStore(), llm=make_stream_llm(chunks))

    events = list(p.ask_stream("学分绩点要求"))

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == "".join(chunks)
    assert events[-1]["mode"] == "kb"
    assert not any(e["type"] == "reset" for e in events)
    assert stub_search == []


def test_ask_stream_short_answer_flushes_after_stream_ends(web_on, stub_search):
    """不足扣留窗口的短回答不能被吞掉。"""
    p = RAGPipeline(StubVectorStore(), llm=make_llm("是的 [1]"))

    events = list(p.ask_stream("学分绩点要求"))

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert deltas == ["是的 [1]"]
    assert events[-1]["mode"] == "kb"


def test_ask_stream_refusal_never_leaks_to_user(web_on, stub_search):
    """扣留窗口内识别出拒答：用户一个字都不该看到「暂未找到」。"""
    p = RAGPipeline(StubVectorStore(), llm=make_llm(KB_REFUSAL, WEB_ANSWER))

    events = list(p.ask_stream("校园网 VPN 怎么用"))

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "暂未找到" not in deltas, "拒答话术必须被扣留，不能闪现"
    assert deltas.startswith(WEB_DISCLAIMER), "网络回答必须由服务端带头注入免责声明"
    assert WEB_ANSWER in deltas
    assert events[-1]["mode"] == "web"

    kinds = [e["type"] for e in events]
    assert "web_sources" in kinds
    assert "status" in kinds
    assert kinds[-1] == "done"


def test_ask_stream_emits_reset_when_refusal_exceeds_holdback(web_on, stub_search):
    """回答已经下发了一部分才判定为拒答时，必须发 reset 让前端清空。"""
    long_refusal = "暂未找到相关依据。" + "知识库中确实没有收录这方面的内容。" * 3
    p = RAGPipeline(
        StubVectorStore(),
        llm=make_stream_llm(["暂未找到相关依据。", "知识库中确实没有收录这方面的内容。" * 3]),
    )
    assert is_unanswerable(long_refusal)

    events = list(p.ask_stream("校园网 VPN 怎么用"))
    kinds = [e["type"] for e in events]

    assert "reset" in kinds, "已下发正文后转联网，必须要求前端清空"
    assert kinds.index("reset") < kinds.index("web_sources")
    assert events[-1]["mode"] == "web"


def test_ask_stream_empty_retrieval_goes_web_without_refusal_flash(web_on, stub_search):
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm(WEB_ANSWER))

    events = list(p.ask_stream("今天长沙天气"))

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert FALLBACK_ANSWER not in deltas
    assert WEB_ANSWER in deltas
    assert events[-1]["mode"] == "web"
    assert stub_search == ["今天长沙天气"]


def test_ask_stream_web_failure_falls_back_to_refusal(web_on, monkeypatch):
    monkeypatch.setattr(
        pipeline_mod, "web_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm())

    events = list(p.ask_stream("今天长沙天气"))

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == FALLBACK_ANSWER
    assert events[-1]["mode"] == "refused"


def test_ask_stream_web_sources_payload_shape(web_on, stub_search):
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm(WEB_ANSWER))

    events = list(p.ask_stream("今天长沙天气"))
    web_event = next(e for e in events if e["type"] == "web_sources")

    assert web_event["sources"] == [
        {
            "file": "湘潭大学",
            "title": "校园网-湘潭大学",
            "url": "https://www.xtu.edu.cn/xysh/ggfw/xyw.htm",
            "snippet": WEB_DOC.content[:160],
            "score": 0.0,
            "kind": "web",
            "publish_date": "",
        }
    ]


# ---------------------------------------------------------------------------
# 多语言：话术必须跟着回答语言走
# ---------------------------------------------------------------------------


def test_english_question_refusal_triggers_web_fallback(web_on, stub_search):
    """英文拒答必须能触发联网兜底。

    真实事故：语言改为「按提问自动识别」后，模型开始用英文拒答，
    而当时的拒答正则是清一色中文 → 识别不出来 → 英文提问**永远**走不到联网兜底。
    """
    english_refusal = fallback_answer("英语")
    p = RAGPipeline(StubVectorStore(), llm=make_llm(english_refusal))

    answer = p.ask("Where can I download MATLAB?", language="英语")

    assert is_unanswerable(english_refusal)
    assert answer.mode == "web"
    assert stub_search == ["Where can I download MATLAB?"]


def test_english_disclaimer_is_injected_in_english(web_on, stub_search):
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm("It is available here [1]."))

    answer = p.ask("Where can I download MATLAB?", language="英语")

    assert answer.text.startswith(web_disclaimer("英语"))
    assert "not officially published" in answer.text
    assert WEB_DISCLAIMER not in answer.text, "英文回答前不该出现中文免责声明"


def test_english_fallback_answer_is_english(stub_search):
    """知识库与网络都没辙时，兜底话术也要是提问者看得懂的语言。"""
    p = RAGPipeline(StubVectorStore(docs=[]), llm=make_llm())

    answer = p.ask("Where can I download MATLAB?", language="英语")

    assert answer.mode == "refused"
    assert answer.text == fallback_answer("英语")
    assert FALLBACK_ANSWER not in answer.text


def test_stream_done_event_reports_language(web_on, stub_search):
    """前端要靠这个字段把来源卡片、结尾提示切到对应语言。"""
    p = RAGPipeline(StubVectorStore(), llm=make_stream_llm(["The GPA ", "threshold is 1.5 [1]."]))

    events = list(p.ask_stream("What is the GPA requirement?", language="英语"))

    assert events[-1]["language"] == "英语"


def test_stream_status_and_reset_are_localised(web_on, stub_search):
    english_refusal = fallback_answer("英语")
    p = RAGPipeline(StubVectorStore(), llm=make_llm(english_refusal, "Web answer [1]."))

    events = list(p.ask_stream("Where can I download MATLAB?", language="英语"))

    status = next(e["text"] for e in events if e["type"] == "status")
    assert status == web_status("英语", len([WEB_DOC]))
    assert "已联网检索" not in status


def test_explicit_language_never_changes_kb_behaviour(web_on, stub_search):
    """语言只影响话术，不该影响「答得出来就绝不联网」这条成本底线。"""
    p = RAGPipeline(StubVectorStore(), llm=make_llm("GPA below 1.5 triggers a warning [1]."))

    answer = p.ask("What is the GPA requirement?", language="英语")

    assert answer.mode == "kb"
    assert stub_search == []


def test_system_prompt_asks_model_to_refuse_in_the_right_language():
    """提示词里那句「照抄」的拒答，必须是提问语言的版本。

    否则模型被要求照抄一句中文——要么中英夹杂，要么自己改写，两条路都糟。
    """
    from xtuagent.rag.pipeline import SYSTEM_PROMPT

    rendered = SYSTEM_PROMPT.format(
        context="（知识库片段）", language="英语", refusal=fallback_answer("英语")
    )
    assert fallback_answer("英语") in rendered
    assert fallback_answer("中文") not in rendered


def test_system_prompt_tells_model_to_translate_not_refuse():
    """知识库是中文的，外文提问不能因此被判「没依据」。

    实测：英文问「What is the GPA warning threshold?」检索分数 0.78（与中文提问同级），
    但模型看到的是中文原文，倾向于回一句拒答 → 白白触发一次付费联网检索。
    这条断言把「照翻译，别说没有」这句话钉在提示词里。
    """
    from xtuagent.rag.pipeline import SYSTEM_PROMPT, WEB_SYSTEM_PROMPT

    kb = SYSTEM_PROMPT.format(context="x", language="英语", refusal="y")
    assert "翻译" in kb
    assert "翻译" in WEB_SYSTEM_PROMPT
