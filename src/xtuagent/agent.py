"""Agent 协作：知识库检索 + 学业查询工具调度。

设计说明：
1. 所有工具的数据来源都是本地知识库（爬取 + 建索引的语料），
   不包含任何硬编码的模拟数据，避免"看起来在查、实际返回假数据"。
2. 工具调用过程中命中的文档会记录到线程局部轨迹中，
   随回答一起返回，便于前端展示引用来源。
3. Agent 实例进程级复用（`get_assistant()`），避免每次请求重建模型客户端。
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.tools import tool

from .rag.vector_store import RetrievedDoc

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
_SNIPPET_LIMIT = 500


class _RetrievalTrace(threading.local):
    """记录单次 Agent 调用中工具命中的检索结果（线程局部，天然隔离并发请求）。"""

    def __init__(self) -> None:
        self.docs: List[RetrievedDoc] = []
        self.seen: set = set()

    def reset(self) -> None:
        self.docs = []
        self.seen = set()

    def add(self, docs: List[RetrievedDoc]) -> None:
        for doc in docs:
            key = (doc.source, doc.content[:80])
            if key in self.seen:
                continue
            self.seen.add(key)
            self.docs.append(doc)


_trace = _RetrievalTrace()


def _get_vector_store():
    from .serving.container import container

    return container.vector_store


def _search(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """统一的工具级检索入口：写回轨迹，返回已按阈值过滤的结果。"""
    store = _get_vector_store()
    if store is None:
        return []
    docs = store.search(query, top_k=top_k)
    _trace.add(docs)
    return docs


def _format(docs: List[RetrievedDoc], empty_hint: str) -> str:
    if not docs:
        return empty_hint
    return "\n\n---\n\n".join(
        f"（来源：{d.source}，相关度 {d.score:.2f}）\n{d.content[:_SNIPPET_LIMIT]}"
        for d in docs
    )


@tool
def search_knowledge_base(query: str) -> str:
    """在校园知识库中检索学业规定、政策或办事指南。用于开放式的信息查找。"""
    if _get_vector_store() is None:
        return "知识库尚未就绪，请稍后再试。"
    docs = _search(query, top_k=3)
    return _format(docs, "知识库中未找到与该问题相关的内容。")


@tool
def course_schedule(course_name: str) -> str:
    """查询某门课程的上课时间、地点与授课教师。请传入课程全称。"""
    docs = _search(f"{course_name} 课程 上课时间 上课地点 授课教师 课程表", top_k=3)
    return _format(
        docs,
        f"知识库中未收录《{course_name}》的课程安排。建议提醒学生核对课程名称，"
        "或登录教务系统查询个人课表。",
    )


@tool
def exam_schedule(course_name: str) -> str:
    """查询某门课程的期末考试时间与考场安排。请传入课程全称。"""
    docs = _search(f"{course_name} 期末考试 考试时间 考场 安排", top_k=3)
    return _format(
        docs,
        f"知识库中未收录《{course_name}》的考试安排。建议提醒学生以教务处"
        "或学院发布的考试通知为准。",
    )


@tool
def regulation_lookup(regulation_name: str) -> str:
    """精确查找某项规定、办法或政策的原文条款。"""
    docs = _search(regulation_name, top_k=1)
    if not docs:
        return f"知识库中未找到关于「{regulation_name}」的原文。"
    doc = docs[0]
    return (
        f"来源：{doc.source}（相关度 {doc.score:.2f}）\n\n"
        f"{doc.content[:800]}\n\n---\n"
        "提示：以上为知识库检索结果，请以学校官方最新文件为准。"
    )


AGENT_TOOLS = [search_knowledge_base, course_schedule, exam_schedule, regulation_lookup]

AGENT_SYSTEM_PROMPT = """你是湘潭大学智能学业助手，可以通过工具查询校园知识库。可用工具：
1. search_knowledge_base —— 检索学业规定、政策、办事指南
2. course_schedule —— 查询课程上课时间与地点
3. exam_schedule —— 查询期末考试安排
4. regulation_lookup —— 查找规定原文条款

工作准则：
- 涉及课程、考试、规定的问题，必须先调用相应工具获取依据，不要凭记忆作答。
- 当工具返回"未找到"类提示时，请如实告知学生该信息暂未收录，
  并建议其向教务处或辅导员确认；不要编造时间、地点或人名。
- 回答简洁准确，用中文回答，并在陈述事实后标注来源（如「来源：xxx」）。"""


@dataclass
class AgentAnswer:
    text: str
    sources: List[RetrievedDoc] = field(default_factory=list)


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
        _trace.reset()
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"recursion_limit": MAX_TOOL_ROUNDS * 2},
            )
        except Exception:  # noqa: BLE001
            logger.exception("Agent 调用失败：%s", user_input)
            return AgentAnswer(
                text="抱歉，本次查询未能完成，请稍后重试或改用普通问答模式。",
                sources=list(_trace.docs),
            )

        text = "无法处理该请求。"
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            if isinstance(content, str) and content.strip():
                text = content
        return AgentAnswer(text=text, sources=list(_trace.docs))


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
