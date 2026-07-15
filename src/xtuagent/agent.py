from langchain_core.tools import tool
from langchain.agents import create_agent


@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索指定内容，返回相关的学业规定、政策或操作指南。"""
    try:
        from .vector_store import load_vector_store
        from .retriever import search_similar

        vs = load_vector_store()
        if vs is None:
            return "知识库尚未构建，请先运行文档入库操作。"
        docs = search_similar(vs, query, top_k=3)
        if not docs:
            return "未找到相关信息。"
        results = [doc.page_content[:500] for doc in docs]
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"知识库搜索失败: {e}"


@tool
def course_schedule(course_name: str) -> str:
    """查询指定课程的上课时间、地点和授课教师。课程名请使用准确的全称。"""
    mock_data = {
        "机器学习": "2024秋季学期：周三 3-4节 (10:00-11:40)，兴湘教学楼C403，王明教授",
        "高等数学": "2024秋季学期：周一 1-2节 (08:00-09:40) 逸夫楼201，张华教授",
        "大学英语": "2024秋季学期：周二 5-6节 (14:00-15:40) 外语楼305，李芳副教授",
        "数据结构": "2024秋季学期：周四 3-4节 (10:00-11:40) 工科楼B101，刘建国教授",
        "马克思主义原理": "2024秋季学期：周五 1-2节 (08:00-09:40) 文科楼102，陈思副教授",
        "大学物理": "2024秋季学期：周一 7-8节 (16:00-17:40) 物理楼201，周磊教授",
        "线性代数": "2024秋季学期：周三 5-6节 (14:00-15:40) 数学楼301，赵婷副教授",
        "数据库原理": "2024秋季学期：周五 3-4节 (10:00-11:40) 工科楼A205，孙伟副教授",
    }
    for key, value in mock_data.items():
        if key in course_name:
            return f"《{key}》：{value}"
    return f"未找到《{course_name}》的课程信息，请确认课程名称是否准确。实际部署时可对接教务系统API获取实时数据。"


@tool
def exam_schedule(course_name: str) -> str:
    """查询指定课程的期末考试时间和地点。"""
    mock_data = {
        "机器学习": "2025年1月8日 14:00-16:00，逸夫楼101",
        "高等数学": "2025年1月6日 08:30-10:30，逸夫楼201",
        "大学英语": "2025年1月7日 14:00-16:00，外语楼多功能厅",
        "数据结构": "2025年1月10日 08:30-10:30，工科楼A101",
        "马克思主义原理": "2025年1月5日 14:00-16:00，文科报告厅",
    }
    for key, value in mock_data.items():
        if key in course_name:
            return f"《{key}》期末考试：{value}"
    return f"未找到《{course_name}》的考试安排，请确认课程名称是否准确。"


@tool
def regulation_lookup(regulation_name: str) -> str:
    """精确查找某条学业规定、校规或政策的原文。"""
    try:
        from .vector_store import load_vector_store
        from .retriever import search_similar

        vs = load_vector_store()
        if vs is None:
            return "知识库尚未构建。"
        docs = search_similar(vs, regulation_name, top_k=1)
        if docs and docs[0].page_content.strip():
            snippet = docs[0].page_content[:800]
            source = docs[0].metadata.get("source_file", "未知来源")
            return f"来源：{source}\n\n{snippet}\n\n---\n提示：以上为知识库检索结果，请以学校官方最新文件为准。"
        return f"未找到关于「{regulation_name}」的规定原文。"
    except Exception as e:
        return f"规定查询失败: {e}"


AGENT_TOOLS = [search_knowledge_base, course_schedule, exam_schedule, regulation_lookup]

AGENT_SYSTEM_PROMPT = """你是湘潭大学智能学业助手，你可以：
1. 从知识库中检索学业规定、政策信息
2. 查询课程上课时间和地点
3. 查询期末考试安排
4. 查询校规和政策原文

当用户问关于课程时间和地点时，主动调用 course_schedule 工具。
当用户问关于考试安排时，主动调用 exam_schedule 工具。
当用户问关于规定和政策时，主动调用 regulation_lookup 或 search_knowledge_base 工具。
回答请简洁、准确，用中文回答。"""


class AgentAssistant:
    def __init__(self):
        from .llm import get_chat_model
        self._model = get_chat_model()
        self._agent = create_agent(
            model=self._model,
            tools=AGENT_TOOLS,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def run(self, user_input: str) -> str:
        result = self._agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
        return "无法处理该请求。"
