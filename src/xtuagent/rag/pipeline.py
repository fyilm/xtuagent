"""RAG 流水线：检索 → 提示词拼装 → 生成（支持流式）。"""

import logging
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..core.config import settings
from ..core.exceptions import LLMInvocationError
from .llm import create_llm
from .retriever import build_context
from .vector_store import RetrievedDoc, VectorStoreService

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员获取更详细的说明。"

SYSTEM_PROMPT = """你是湘潭大学（Xiangtan University）的智能学业助手，专门为学生提供准确的学业信息。

请根据以下知识库内容回答问题。回答要求：
1. 如果知识库中有相关信息，请准确、简洁地回答，必要时分点列出。
2. 如果知识库中没有相关信息，请如实告知"根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员"。
3. 涉及政策规定时，请注明信息来源（如"根据《湘潭大学学生手册》……"）。
4. 回答使用{language}语言。

知识库内容：
---
{context}
---"""


@dataclass
class Answer:
    text: str
    sources: List[RetrievedDoc]
    elapsed_ms: int


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreService,
        llm=None,
        top_k: Optional[int] = None,
    ) -> None:
        self._vs = vector_store
        self._llm = llm
        self._top_k = top_k or settings.retriever_top_k

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm()
        return self._llm

    def retrieve(self, question: str) -> List[RetrievedDoc]:
        return self._vs.search(question, self._top_k)

    def _build_chain(self, context: str, language: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.format(context=context, language=language)),
                ("human", "{question}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _invoke_with_retry(self, chain, question: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                return chain.invoke({"question": question})
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait = attempt * 3
                logger.warning("LLM 调用失败（第 %d 次）：%s，%ds 后重试", attempt, exc, wait)
                time.sleep(wait)
        raise LLMInvocationError(str(last_error))

    def ask(self, question: str, language: str = "中文") -> Answer:
        start = time.time()
        docs = self.retrieve(question)
        if not docs:
            return Answer(
                text=FALLBACK_ANSWER,
                sources=[],
                elapsed_ms=int((time.time() - start) * 1000),
            )

        chain = self._build_chain(build_context(docs), language)
        text = self._invoke_with_retry(chain, question)
        return Answer(
            text=text,
            sources=docs,
            elapsed_ms=int((time.time() - start) * 1000),
        )

    def ask_stream(self, question: str, language: str = "中文") -> Iterator[dict]:
        """流式问答：依次产出 sources / delta / done 事件。"""
        start = time.time()
        docs = self.retrieve(question)
        yield {
            "type": "sources",
            "sources": [
                {"file": d.source, "snippet": d.content[:160], "score": d.score}
                for d in docs
            ],
        }

        if not docs:
            yield {"type": "delta", "text": FALLBACK_ANSWER}
            yield {"type": "done", "elapsed_ms": int((time.time() - start) * 1000)}
            return

        chain = self._build_chain(build_context(docs), language)
        received = False
        try:
            for chunk in chain.stream({"question": question}):
                if chunk:
                    received = True
                    yield {"type": "delta", "text": chunk}
        except Exception as exc:  # noqa: BLE001
            if received:
                logger.warning("流式输出中断：%s", exc)
            else:
                logger.warning("流式调用失败，降级为同步调用：%s", exc)
                text = self._invoke_with_retry(chain, question)
                yield {"type": "delta", "text": text}

        yield {"type": "done", "elapsed_ms": int((time.time() - start) * 1000)}
