"""Agent 协作：知识库检索 + 学业查询工具调度 + 联网兜底。

设计说明：
1. 所有工具的数据来源都是本地知识库（爬取 + 建索引的语料），
   唯一的例外是 `web_lookup`——它在知识库明确没有依据时才去互联网检索，
   返回内容会标注为网络来源，不与校内语料混淆。
2. 工具调用过程中命中的文档会记录到**本次请求的轨迹**中，随回答一起返回，
   便于前端展示引用来源。轨迹通过 `RunnableConfig.configurable` 传递，见下。
3. Agent 实例进程级复用（`get_assistant()`），避免每次请求重建模型客户端。

## 为什么轨迹不能用 threading.local 存

历史实现用的是 `threading.local` 子类，理由是"天然隔离并发请求"。但它有个致命假设：
**工具函数与 `run()` 跑在同一个线程**。实测这个假设不成立——
LangGraph 的 ToolNode 会把工具丢进 worker 线程执行（实测工具线程 ≠ run 线程），
于是工具写进去的轨迹在 `run()` 里读出来永远是空的：

    [add_web] thread=18908 docs=5      ← 工具在 worker 线程写入
    run 侧最终 docs=0 web_docs=0       ← run() 在主线程读取

表现为「工具增强模式永远显示 0 条来源」，且不影响答案本身，所以很难被发现。

现在改为把每次请求的轨迹对象放进 `config.configurable`，由 LangChain 自动注入
给声明了 `config: RunnableConfig` 参数的工具。既跨线程可见，又天然按请求隔离，
并且不需要加锁。
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from .core.config import settings
from .rag.pipeline import WEB_DISCLAIMER, is_unanswerable
from .rag.vector_store import RetrievedDoc
from .rag.websearch import WebDoc
from .rag.websearch import web_search as _web_search

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
_SNIPPET_LIMIT = 500

# 轨迹对象在 RunnableConfig 里的键名
TRACE_KEY = "xtuagent_trace"


class _RetrievalTrace:
    """单次 Agent 调用的检索轨迹。普通对象即可，线程安全由「每请求一个」保证。"""

    def __init__(self) -> None:
        self.docs: List[RetrievedDoc] = []
        self.web_docs: List[WebDoc] = []
        self.seen: set = set()

    def add(self, docs: List[RetrievedDoc]) -> None:
        for doc in docs:
            key = (doc.source, doc.content[:80])
            if key in self.seen:
                continue
            self.seen.add(key)
            self.docs.append(doc)

    def add_web(self, docs: List[WebDoc]) -> None:
        for doc in docs:
            key = ("web", doc.link or doc.title, doc.content[:80])
            if key in self.seen:
                continue
            self.seen.add(key)
            self.web_docs.append(doc)


# 直接调用工具（不经 Agent，例如单元测试）时的兜底轨迹
_fallback_trace = _RetrievalTrace()


def _trace_from(config: Optional[RunnableConfig]) -> _RetrievalTrace:
    if config:
        trace = (config.get("configurable") or {}).get(TRACE_KEY)
        if isinstance(trace, _RetrievalTrace):
            return trace
    return _fallback_trace


def _get_vector_store():
    from .serving.container import container

    return container.vector_store


def _search(query: str, top_k: int, config: Optional[RunnableConfig]) -> List[RetrievedDoc]:
    """统一的工具级检索入口：写回轨迹，返回已按阈值过滤的结果。"""
    store = _get_vector_store()
    if store is None:
        return []
    docs = store.search(query, top_k=top_k)
    _trace_from(config).add(docs)
    return docs


def _format(docs: List[RetrievedDoc], empty_hint: str) -> str:
    if not docs:
        return empty_hint
    return "\n\n---\n\n".join(
        f"（来源：{d.source}，相关度 {d.score:.2f}）\n{d.content[:_SNIPPET_LIMIT]}"
        for d in docs
    )


@tool
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """在校园知识库中检索学业规定、政策或办事指南。用于开放式的信息查找。"""
    if _get_vector_store() is None:
        return "知识库尚未就绪，请稍后再试。"
    docs = _search(query, top_k=3, config=config)
    return _format(docs, "知识库中未找到与该问题相关的内容。")


@tool
def course_schedule(course_name: str, config: RunnableConfig) -> str:
    """查询某门课程的上课时间、地点与授课教师。请传入课程全称。"""
    docs = _search(
        f"{course_name} 课程 上课时间 上课地点 授课教师 课程表", top_k=3, config=config
    )
    return _format(
        docs,
        f"知识库中未收录《{course_name}》的课程安排。建议提醒学生核对课程名称，"
        "或登录教务系统查询个人课表。",
    )


@tool
def exam_schedule(course_name: str, config: RunnableConfig) -> str:
    """查询某门课程的期末考试时间与考场安排。请传入课程全称。"""
    docs = _search(
        f"{course_name} 期末考试 考试时间 考场 安排", top_k=3, config=config
    )
    return _format(
        docs,
        f"知识库中未收录《{course_name}》的考试安排。建议提醒学生以教务处"
        "或学院发布的考试通知为准。",
    )


@tool
def regulation_lookup(regulation_name: str, config: RunnableConfig) -> str:
    """精确查找某项规定、办法或政策的原文条款。"""
    docs = _search(regulation_name, top_k=1, config=config)
    if not docs:
        return f"知识库中未找到关于「{regulation_name}」的原文。"
    doc = docs[0]
    return (
        f"来源：{doc.source}（相关度 {doc.score:.2f}）\n\n"
        f"{doc.content[:800]}\n\n---\n"
        "提示：以上为知识库检索结果，请以学校官方最新文件为准。"
    )


@tool
def web_lookup(query: str, config: RunnableConfig) -> str:
    """当知识库明确没有相关内容时，到互联网上检索公开资料来回答。

    仅在 search_knowledge_base 返回「未找到」之后才使用。传入精简的检索关键词，
    不要直接把学生口语化的整句话丢进去（例如传「湘潭大学 校园网 VPN 使用方法」，
    而不是「我们学校那个vpn到底咋弄啊」）。
    """
    if not settings.web_fallback_enabled:
        return "联网检索当前已关闭。请如实告知学生该信息暂未收录。"
    try:
        docs = _web_search(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Agent 联网检索失败：%s", exc)
        return "联网检索暂时不可用，请如实告知学生该信息暂未收录，并建议其咨询相关部门。"
    if not docs:
        return "互联网上也没有检索到与该问题相关的可靠资料。请如实告知学生。"
    _trace_from(config).add_web(docs)
    return (
        "以下内容来自互联网公开网页，**不是学校官方发布**，回答时必须先向学生说明这一点。\n\n"
        + "\n\n---\n\n".join(
            f"（来源：{d.label}｜{d.title[:60]}｜{d.link}）\n{d.content[:_SNIPPET_LIMIT]}"
            for d in docs
        )
    )


AGENT_TOOLS = [
    search_knowledge_base,
    course_schedule,
    exam_schedule,
    regulation_lookup,
    web_lookup,
]

AGENT_SYSTEM_PROMPT = """你是湘潭大学智能学业助手，可以通过工具查询校园知识库。可用工具：
1. search_knowledge_base —— 检索学业规定、政策、办事指南（**优先使用**）
2. course_schedule —— 查询课程上课时间与地点
3. exam_schedule —— 查询期末考试安排
4. regulation_lookup —— 查找规定原文条款
5. web_lookup —— 知识库没有依据时，到互联网检索公开资料

工作准则：
- 涉及课程、考试、规定的问题，必须先调用相应工具获取依据，不要凭记忆作答。
- 当知识库工具返回"未找到"类提示时，**不要就此打住**：改用 web_lookup 到互联网再找一次，
  然后把查到的内容整理给学生。
- 使用 web_lookup 的结果时，必须明确告诉学生"以下信息来自网络、非学校官方口径，
  请以学校官方通知为准"，并给出可点击的来源网址。
- 只有知识库和互联网都没有找到时，才如实告知学生"暂未收录"，并建议其向教务处、
  网络与信息管理中心或辅导员确认。任何情况下都不要编造时间、地点、网址或人名。
- 回答简洁准确，用中文回答，并在陈述事实后标注来源（如「来源：xxx」）。"""



@dataclass
class AgentAnswer:
    text: str
    sources: List[RetrievedDoc] = field(default_factory=list)
    web_sources: List[WebDoc] = field(default_factory=list)


class AgentAssistant:
    """基于 ReAct Agent 的智能助手（工具调用）。"""

    def __init__(self) -> None:
        from langchain.agents import create_agent

        from .rag.llm import create_llm

        self._agent = create_agent(
            model=create_llm(),
            tools=AGENT_TOOLS,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def run(self, user_input: str) -> AgentAnswer:
        # 每次请求新建一个轨迹，放进 config 里随调用链下传。
        # 工具可能在 worker 线程执行，但 config 会被 LangChain 一并带过去。
        trace = _RetrievalTrace()
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={
                    "recursion_limit": MAX_TOOL_ROUNDS * 2,
                    "configurable": {TRACE_KEY: trace},
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("Agent 调用失败：%s", user_input)
            return AgentAnswer(
                text="抱歉，本次查询未能完成，请稍后重试或改用普通问答模式。",
                sources=list(trace.docs),
                web_sources=list(trace.web_docs),
            )

        text = "无法处理该请求。"
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            if isinstance(content, str) and content.strip():
                text = content

        web_docs = list(trace.web_docs)
        if not web_docs and is_unanswerable(text):
            # 模型只说"没收录"却没去联网 —— 不能就这么把用户打发了。
            # 提示词里虽然写了"改用 web_lookup"，但那是**请求**不是**保证**，
            # 所以这里补一道确定性兜底：直接调用流水线的联网作答。
            fallback = self._web_fallback(user_input)
            if fallback is not None:
                return fallback

        if web_docs:
            # 与普通问答一致：只要联网资料被读过，就由服务端强制打上来源声明，
            # 不指望模型自己记得写（实测它会漏）。
            text = WEB_DISCLAIMER + text

        return AgentAnswer(text=text, sources=list(trace.docs), web_sources=web_docs)

    @staticmethod
    def _web_fallback(user_input: str) -> Optional[AgentAnswer]:
        from .serving.container import container

        pipeline = getattr(container, "pipeline", None)
        if pipeline is None:
            return None
        logger.info("Agent 未联网即判定未收录，补一次确定性联网兜底：%s", user_input)
        answer = pipeline.answer_with_web(user_input)
        if answer is None:
            return None
        return AgentAnswer(
            text=answer.text,
            sources=[],
            web_sources=list(answer.web_sources),
        )


_assistant: Optional[AgentAssistant] = None
_assistant_lock = threading.Lock()


def get_assistant() -> AgentAssistant:
    """进程级单例。Agent 构建涉及 LLM 客户端初始化，不应每个请求重建。"""
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                _assistant = AgentAssistant()
                logger.info("Agent 已初始化：%d 个工具", len(AGENT_TOOLS))
    return _assistant
