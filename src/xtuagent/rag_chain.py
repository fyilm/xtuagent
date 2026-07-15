import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import config, LLMProvider
from .retriever import search_similar
from .llm import get_chat_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是湘潭大学(Xiangtan University)的智能学业助手，专门为在校学生提供准确的学业相关信息。

请根据以下知识库内容回答问题。回答要求：
1. 如果知识库中有相关信息，请准确、简洁地回答，必要时分点列出。
2. 如果知识库中没有相关信息，请如实告知"根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员"。
3. 涉及政策规定时，请注明信息来源（如"根据《湘潭大学学生手册》..."）。
4. 回答使用{language}语言。

知识库内容：
---
{context}
---"""


class RAGChain:
    def __init__(
        self,
        vector_store: Chroma,
        provider: Optional[LLMProvider] = None,
        top_k: Optional[int] = None,
    ):
        self.vector_store = vector_store
        self._provider = provider
        self._model = None
        self.top_k = top_k or config.retriever_top_k

    @property
    def model(self):
        if self._model is None:
            self._model = get_chat_model(provider=self._provider)
        return self._model

    def ask(self, question: str, language: str = "中文") -> str:
        docs = search_similar(self.vector_store, question, self.top_k)
        if not docs:
            return "根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员。"

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        system_prompt = SYSTEM_PROMPT.format(context=context, language=language)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        chain = prompt | self.model | StrOutputParser()

        for attempt in range(5):
            try:
                return chain.invoke({"question": question})
            except Exception as e:
                if attempt == 4:
                    logger.error("LLM generation failed after 5 attempts: %s", e)
                    return f"回答生成失败: {e}"
                wait = (attempt + 1) * 5
                import time
                logger.warning("LLM attempt %d failed, retry in %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)

    def get_relevant_sources(self, question: str) -> list[str]:
        docs = search_similar(self.vector_store, question, self.top_k)
        sources = []
        for doc in docs:
            source = doc.metadata.get("source_file", "unknown")
            if source not in sources:
                sources.append(source)
        return sources
