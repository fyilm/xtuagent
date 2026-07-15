import gradio as gr
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xtuagent.config import config, LLMProvider
from xtuagent.vector_store import load_vector_store
from xtuagent.rag_chain import RAGChain
from xtuagent.agent import AgentAssistant

logging.basicConfig(level=logging.INFO)

CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
footer { display: none !important; }
"""


def ask_rag(
    question: str,
    provider_label: str,
    mode_label: str,
    language: str,
    chat_history: list,
) -> str:
    if not question.strip():
        return "请输入问题。"

    provider_map = {
        "智谱 GLM-4 (中文推荐)": "zhipu",
        "文心一言 (外语推荐 / Recommended for English)": "qianfan",
    }
    lang_map = {"中文": "中文", "English": "English", "Français": "French", "Deutsch": "German"}
    provider: LLMProvider = provider_map.get(provider_label, "zhipu")
    answer_lang = lang_map.get(language, "中文")

    try:
        vs = load_vector_store()
        if vs is None:
            return "知识库尚未构建，请先运行爬虫和文档入库操作。"

        if "agent" in mode_label.lower():
            assistant = AgentAssistant()
            return assistant.run(question)
        else:
            rag = RAGChain(vector_store=vs, provider=provider)
            return rag.ask(question, language=answer_lang)
    except Exception as e:
        return f"系统错误: {e}"


demo = gr.ChatInterface(
    fn=ask_rag,
    title="📚 湘潭大学智能学业问答助手",
    description="基于RAG架构的私域知识库问答系统",
    additional_inputs=[
        gr.Dropdown(
            ["智谱 GLM-4 (中文推荐)", "文心一言 (外语推荐 / Recommended for English)"],
            value="智谱 GLM-4 (中文推荐)",
            label="LLM 引擎",
        ),
        gr.Dropdown(
            ["简单问答 (RAG)", "智能助手 (Agent)"],
            value="简单问答 (RAG)",
            label="模式",
        ),
        gr.Dropdown(
            ["中文", "English", "Français", "Deutsch"],
            value="中文",
            label="回答语言",
        ),
    ],
    css=CSS,
)

if __name__ == "__main__":
    demo.launch()
