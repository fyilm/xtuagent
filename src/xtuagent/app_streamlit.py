import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from xtuagent.config import config, LLMProvider
from xtuagent.vector_store import load_vector_store
from xtuagent.rag_chain import RAGChain

st.set_page_config(page_title="湘潭大学学业助手", page_icon="📚", layout="wide")
st.title("📚 湘潭大学智能学业问答助手")
st.caption("基于RAG架构的私域知识库问答系统")

LLM_MAP = {
    "智谱 GLM-4-Flash (默认)": "zhipu",
    "文心一言 ERNIE-Lite": "qianfan",
}

with st.sidebar:
    st.header("⚙️ 设置")
    provider_label = st.selectbox(
        "LLM 引擎", list(LLM_MAP.keys()), index=0
    )
    provider: LLMProvider = LLM_MAP[provider_label]
    language = st.selectbox("回答语言", ["中文", "English"], index=0)
    if st.button("🔄 重新连接知识库"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.divider()

if "vs" not in st.session_state:
    with st.spinner("正在连接知识库..."):
        try:
            st.session_state.vs = load_vector_store()
            if st.session_state.vs is None:
                st.error("知识库未找到")
            else:
                st.success(f"知识库就绪 ({st.session_state.vs._collection.count()} 条记录)")
        except Exception as e:
            st.error(f"知识库加载失败: {e}")

if "rag" not in st.session_state and "vs" in st.session_state and st.session_state.vs:
    st.session_state.rag = RAGChain(vector_store=st.session_state.vs, provider="zhipu")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("请输入你的学业问题..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # Recreate RAG if provider changed
                if "last_provider" not in st.session_state or st.session_state.last_provider != provider:
                    st.session_state.rag = RAGChain(
                        vector_store=st.session_state.vs, provider=provider
                    )
                    st.session_state.last_provider = provider

                lang = "中文" if language == "中文" else "English"
                answer = st.session_state.rag.ask(question, language=lang)
            except Exception as e:
                answer = f"系统错误: {type(e).__name__}: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
