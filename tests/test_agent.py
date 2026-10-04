"""Agent（工具增强模式）测试。

为什么单独补这个文件：`agent.py` 此前**零测试覆盖**，219 项全绿的测试套件
完全没发现 `run()` 里的 `self._agent` 是个不存在的属性——
那是我把构造改成「按语言缓存 Agent」时改漏的一个调用点，
表现是工具增强模式**每一个请求都立刻返回「抱歉，本次查询未能完成」**，
而且因为异常被 `except Exception` 兜住了，日志之外没有任何征兆。

本文件全部使用桩 Agent，不打 LLM、不打搜索引擎、也不加载嵌入模型。
"""

import threading

import pytest

from xtuagent.agent import (
    DEFAULT_LANGUAGE,
    MAX_TOOL_ROUNDS,
    TRACE_KEY,
    AgentAnswer,
    AgentAssistant,
    _RetrievalTrace,
)
from xtuagent.core.messages import fallback_answer, service_error, unhandled, web_disclaimer
from xtuagent.rag.vector_store import RetrievedDoc
from xtuagent.rag.websearch import WebDoc

KB_DOC = RetrievedDoc(content="转专业需提交申请表。", source="学籍管理规定.txt", score=0.81)
WEB_DOC = WebDoc(
    title="校园网-湘潭大学",
    link="https://www.xtu.edu.cn/xw.htm",
    media="湘潭大学",
    content="校园网提供免费 VPN。",
)


class _Msg:
    def __init__(self, content) -> None:
        self.content = content


class StubAgent:
    """记录调用参数的桩 Agent。"""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list = []

    def invoke(self, payload, config=None):
        self.calls.append({"payload": payload, "config": config})
        if self.raises is not None:
            raise self.raises
        return self.result


def make_assistant(agent: StubAgent, *, trace=None) -> AgentAssistant:
    """绕开 __init__ 的重活（导入 create_agent / 建 LLM 客户端），只装必需的属性。"""
    a = AgentAssistant.__new__(AgentAssistant)
    a._agents = {}
    a._agents_lock = threading.Lock()
    a._model = object()

    def _factory(**kwargs):
        if trace is not None:
            trace.append(kwargs)
        return agent

    a._create_agent = _factory
    return a


def reply(text: str) -> dict:
    return {"messages": [_Msg(text)]}


# ---------------------------------------------------------------------------
# 回归：run() 必须走「按语言取 Agent」，不能再碰 self._agent
# ---------------------------------------------------------------------------


def test_run_invokes_the_agent():
    """回归用例：曾经写成 `self._agent.invoke(...)`，属性不存在 → 每个请求都失败。

    这条断言本身很朴素，但它挡住的是「工具增强模式整体不可用」这种级别的故障。
    """
    agent = StubAgent(result=reply("转专业需提交申请表 [1]。"))
    assistant = make_assistant(agent)

    answer = assistant.run("转专业需要什么条件？")

    assert len(agent.calls) == 1, "run() 没有真正把请求交给 Agent"
    assert answer.text == "转专业需提交申请表 [1]。"


def test_no_leftover_single_agent_attribute():
    """实例上不该再有单例式的 `_agent` 属性——那是改按语言缓存前的残留。"""
    assistant = make_assistant(StubAgent(result=reply("x")))
    assistant.run("你好")
    assert not hasattr(assistant, "_agent")


def test_run_passes_trace_and_recursion_limit():
    agent = StubAgent(result=reply("ok"))
    assistant = make_assistant(agent)

    assistant.run("你好")

    config = agent.calls[0]["config"]
    assert isinstance(config["configurable"][TRACE_KEY], _RetrievalTrace)
    assert config["recursion_limit"] == MAX_TOOL_ROUNDS * 2
    assert agent.calls[0]["payload"]["messages"][0]["content"] == "你好"


# ---------------------------------------------------------------------------
# 语言：Agent 的系统提示词是按语言构建的
# ---------------------------------------------------------------------------


def test_agent_is_built_per_language_and_cached():
    built = []
    agent = StubAgent(result=reply("ok"))
    assistant = make_assistant(agent, trace=built)

    assistant.run("转专业条件", language="英语")
    assistant.run("转专业条件", language="英语")
    assistant.run("转专业条件", language="日语")

    assert [k["system_prompt"] for k in built] and len(built) == 2, "同语言应复用、异语言才重建"
    assert len(assistant._agents) == 2


@pytest.mark.parametrize("language", ["中文", "英语", "日语", "韩语", "俄语"])
def test_agent_prompt_carries_language_specific_copy(language):
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for(language)

    prompt = built[0]["system_prompt"]
    assert f"使用{language}语言" in prompt
    assert fallback_answer(language) in prompt, "拒答例句必须是提问语言的版本"
    assert "{" not in prompt and "}" not in prompt, "模板占位符必须全部被替换掉"


def test_agent_prompt_does_not_ask_model_to_write_the_disclaimer():
    """声明由服务端注入，提示词里要让模型别再写一遍，否则会出现两句免责声明。"""
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for("英语")

    prompt = built[0]["system_prompt"]
    assert "不要自己写" in prompt
    assert web_disclaimer("英语").strip() in prompt


# ---------------------------------------------------------------------------
# 失败与兜底：消息必须跟着语言走
# ---------------------------------------------------------------------------


def test_agent_failure_returns_localized_service_error():
    agent = StubAgent(raises=RuntimeError("boom"))
    assistant = make_assistant(agent)

    answer = assistant.run("你好", language="日语")

    assert answer.text == service_error("日语")
    assert answer.text != service_error(DEFAULT_LANGUAGE)


def test_empty_result_returns_localized_placeholder():
    assistant = make_assistant(StubAgent(result={"messages": []}))

    answer = assistant.run("你好", language="英语")

    assert answer.text == unhandled("英语")


def test_refusal_without_web_docs_triggers_deterministic_web_fallback(monkeypatch):
    """模型只说「没收录」却忘了联网时，补一次确定性联网兜底。"""
    agent = StubAgent(result=reply(fallback_answer("英语")))
    assistant = make_assistant(agent)
    sentinel = AgentAnswer(text="from web", web_sources=[WEB_DOC])

    monkeypatch.setattr(
        AgentAssistant, "_web_fallback", staticmethod(lambda q, lang: sentinel)
    )

    answer = assistant.run("Where can I download MATLAB?", language="英语")

    assert answer is sentinel


def test_normal_answer_never_triggers_web_fallback(monkeypatch):
    """能答的问题不许产生搜索费用——成本底线。"""
    agent = StubAgent(result=reply("平均学分绩点低于 1.5 会收到学业警告 [1]。"))
    assistant = make_assistant(agent)

    def _boom(*_a, **_k):
        raise AssertionError("不该调用联网兜底")

    monkeypatch.setattr(AgentAssistant, "_web_fallback", staticmethod(_boom))

    answer = assistant.run("学分绩点要求")

    assert answer.text.startswith("平均学分绩点")


def test_web_docs_get_localized_disclaimer():
    """读过联网资料就必须由服务端打上来源声明，且与回答同语言。"""
    agent = StubAgent(result=reply("VPN 需先登录学校邮箱 [1]。"))
    assistant = make_assistant(agent)

    # 真实场景里轨迹由工具函数写入；这里让桩在 invoke 时模拟一次联网命中
    original_invoke = agent.invoke

    def _invoke(payload, config=None):
        config["configurable"][TRACE_KEY].add_web([WEB_DOC])
        return original_invoke(payload, config)

    agent.invoke = _invoke  # type: ignore[method-assign]

    answer = assistant.run("VPN 怎么用", language="英语")

    assert answer.text == web_disclaimer("英语") + "VPN 需先登录学校邮箱 [1]。"
    assert answer.web_sources and answer.web_sources[0].link == WEB_DOC.link
    assert "非学校官方发布" not in answer.text, "英文回答不该被套上中文免责声明"


def test_kb_docs_are_returned_without_disclaimer():
    """没读过联网资料时不许加免责声明——那会让人以为答案来自网上。"""
    agent = StubAgent(result=reply("转专业需提交申请表 [1]。"))
    assistant = make_assistant(agent)
    original_invoke = agent.invoke

    def _invoke(payload, config=None):
        config["configurable"][TRACE_KEY].add([KB_DOC])
        return original_invoke(payload, config)

    agent.invoke = _invoke  # type: ignore[method-assign]

    answer = assistant.run("转专业条件")

    assert answer.text == "转专业需提交申请表 [1]。"
    assert answer.sources == [KB_DOC]
    assert answer.web_sources == []


# ---------------------------------------------------------------------------
# web_lookup 工具：只给资料，不给「对模型说的话」
# ---------------------------------------------------------------------------


def test_web_lookup_does_not_leak_meta_instructions(monkeypatch):
    """工具返回值里不能夹带写给模型的指令。

    实测（2026-10-02）：工具原本会返回「以下内容来自互联网公开网页，不是学校官方发布，
    回答时必须先向学生说明这一点。」——glm-4-flash 把这句原样抄进了最终回答，
    学生看到一句对模型说的话；而且服务端还会再注入一次免责声明，变成两句。
    """
    from xtuagent import agent as agent_mod
    from xtuagent.agent import web_lookup

    monkeypatch.setattr(agent_mod.settings, "web_fallback_enabled", True)
    monkeypatch.setattr(agent_mod, "_web_search", lambda q: [WEB_DOC])
    trace = _RetrievalTrace()

    out = web_lookup.func("校园网 VPN 使用方法", {"configurable": {TRACE_KEY: trace}})

    assert "回答时必须先向学生说明" not in out
    assert "不是学校官方发布" not in out
    assert WEB_DOC.link in out and WEB_DOC.title[:10] in out
    assert trace.web_docs == [WEB_DOC]


def test_web_lookup_degrades_gracefully(monkeypatch):
    """联网不可用时返回可读提示，而不是抛异常把整轮对话搞崩。"""
    from xtuagent import agent as agent_mod
    from xtuagent.agent import web_lookup

    monkeypatch.setattr(agent_mod.settings, "web_fallback_enabled", True)
    monkeypatch.setattr(
        agent_mod, "_web_search", lambda q: (_ for _ in ()).throw(RuntimeError("429"))
    )

    out = web_lookup.func("随便", {"configurable": {TRACE_KEY: _RetrievalTrace()}})

    assert "暂未收录" in out or "不可用" in out


def test_web_lookup_respects_the_kill_switch(monkeypatch):
    from xtuagent import agent as agent_mod
    from xtuagent.agent import web_lookup

    monkeypatch.setattr(agent_mod.settings, "web_fallback_enabled", False)
    called = []
    monkeypatch.setattr(agent_mod, "_web_search", lambda q: called.append(q) or [WEB_DOC])

    out = web_lookup.func("随便", {"configurable": {TRACE_KEY: _RetrievalTrace()}})

    assert called == [], "开关关闭时不该产生搜索费用"
    assert "关闭" in out


# ---------------------------------------------------------------------------
# 提示词：把实测踩过的坑钉住
# ---------------------------------------------------------------------------


def test_agent_prompt_equates_irrelevant_results_with_not_found():
    """「搜到一堆无关内容」必须被当作「没找到」，否则模型会拿无关片段凑答案。

    实测：「什么是摩尔定律？」检索到的是《毛泽东研究动态》，模型就把这些罗列出来了，
    既没回答问题，也没走上网兜底。
    """
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for("中文")
    prompt = built[0]["system_prompt"]

    assert "无关" in prompt and "等同于" in prompt
    assert "web_lookup" in prompt


def test_agent_prompt_allows_non_academic_questions():
    """寒暄、纯计算这类问题不该被逼着去检索知识库。"""
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for("中文")

    assert "1+1" in built[0]["system_prompt"]


def test_agent_prompt_forbids_answering_facts_from_memory():
    """「与学业无关就直接答」这条曾经被误用成「天气也可以凭记忆答」。

    实测（2026-10-02）：加了「与学业无关的问题照常直接回答」之后，
    问天气时模型答「这个问题与学业无关，我可以直接回答」，
    然后凭记忆报了一个**错误的**气温。所以口径必须收窄：
    寒暄/纯计算可直答，**需要外部事实的一律先 web_lookup 查证**。
    """
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for("中文")
    prompt = built[0]["system_prompt"]

    assert "需要外部事实" in prompt
    assert "天气" in prompt
    assert "绝不许凭记忆作答" in prompt


def test_agent_prompt_forbids_claiming_a_web_search_it_did_not_do():
    """模型会「声称联网」却没真的调工具，还贴一个凭记忆编出来的链接。

    实测：「什么是摩尔定律？」的 Agent 回答写着「以下是我从互联网上找到的更详细的信息」
    并附维基链接，但本次轨迹里 web_docs 为空——它根本没调用 web_lookup。
    """
    built = []
    assistant = make_assistant(StubAgent(result=reply("ok")), trace=built)

    assistant._agent_for("中文")
    prompt = built[0]["system_prompt"]

    assert "没调用过 web_lookup 就不要说" in prompt
