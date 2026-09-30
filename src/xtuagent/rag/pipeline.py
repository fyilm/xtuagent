"""RAG 流水线：检索 → 提示词拼装 → 生成（支持流式），知识库答不出来时自动联网兜底。

## 两层回答策略

1. **知识库优先**：检索 → 拼上下文 → 生成。这是常规路径，不产生搜索费用。
2. **联网兜底**：当模型读完片段明确表示「答不了」时，立刻转向互联网检索，
   用网络资料重新作答，并在回答里标明「来自网络、非官方」。

关键设计取舍——**什么算「答不了」**：

检索层不做相关性过滤（阈值默认 0，原因见 `core/config.py`），所以「检索到 5 条」
并不等于「有依据」。真正可靠的信号只有一个：**模型读完之后说答不了**。
因此拒答识别被刻意做得**保守**——宁可漏判（保持原来的拒答表现），
不可误判（把本来答得好好的内容换成网络答案）。
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..core.config import settings
from ..core.exceptions import LLMInvocationError
from .llm import create_llm
from .retriever import build_context
from .vector_store import RetrievedDoc, VectorStoreService
from .websearch import WebDoc, build_web_context, web_search

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员获取更详细的说明。"

SYSTEM_PROMPT = """你是湘潭大学（Xiangtan University）的智能学业助手，专门为学生提供准确的学业信息。

请严格依据下面提供的知识库内容回答问题。回答要求：
1. 只使用知识库中出现的信息。不要依赖你自己的先验知识补充细节（尤其是数字、日期、比例、人名）。
2. 如果知识库中没有足够信息回答该问题，请如实回复"根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员获取更详细的说明"，不要猜测。
3. 回答准确、简洁，必要时分点列出，并在句末用 [编号] 标注信息来源（例如"平均学分绩点低于 1.5 会收到学业警告 [1]"）。
4. 回答使用{language}语言。

知识库内容：
---
{context}
---"""

WEB_SYSTEM_PROMPT = """你是湘潭大学（Xiangtan University）的智能学业助手。学生的问题在校内知识库中
没有找到依据，下面是刚刚从互联网检索到的资料，请据此作答。

回答要求：
1. 关于"内容来自网络"的声明由系统自动加在回答最前面，**你不需要再写一遍**，
   直接进入正文即可。
2. 只使用下面资料里的信息，不要凭记忆补充。资料里没有的关键信息，直接说没有查到。
3. 涉及时间、地点、金额、网址、电话、密码规则等关键信息时，务必逐字核对后再写。
4. 引用资料时在句末标注编号，例如"VPN 客户端需先登录学校邮箱再下载 [2]"。
5. 如果资料与问题无关或明显不足以回答，请如实说明网上也没有查到，并建议用户
   咨询网络与信息管理中心（网络类问题）或教务处、辅导员（学业类问题）。
6. 回答使用{language}语言，准确、简洁，必要时分点列出。

网络检索资料：
---
{context}
---"""

# 联网回答的免责声明。
#
# **由服务端强制注入，不交给模型**。原因：实测 glm-4-flash 会漏掉提示词里
# "必须先说明来源性质"这条要求，直接开讲天气——而"这条信息到底是不是学校官方口径"
# 恰恰是用户最不能搞错的一件事。凡是不能容忍被跳过的内容，都不该指望模型自觉。
WEB_DISCLAIMER = "**以下内容来自网络检索，非学校官方发布，请以学校官方通知为准。**\n\n"

# ---------------------------------------------------------------------------
# 拒答识别
# ---------------------------------------------------------------------------

# 只在回答**开头**这段范围内判定。正文中偶然出现「未收录」不该被当成整条拒答，
# 例如「学分绩点低于 1.5 会收到学业警告。如果没有找到相关豁免条款，请联系教务处」
# 是一句完整的正常回答，绝不能被误判。
_REFUSAL_SCAN_CHARS = 30

# 全部锚定在回答开头，且只允许极短的引导语（"很抱歉，"这类）。
# 误判的代价是「把好答案换成网络答案」，比漏判严重得多，所以这里刻意保守。
_REFUSAL_PATTERNS = (
    # 提示词给定的原句开头，模型改写后半句也会保留它
    re.compile(r"^根据现有知识库"),
    re.compile(r"^.{0,10}暂未找到相关"),
    re.compile(r"^.{0,10}(未能|无法|没有|不能)找到相关"),
    re.compile(r"^.{0,10}(暂未|尚未|未)收录"),
    re.compile(r"^.{0,10}知识库(中|里)?.{0,6}(暂未|尚未|未|没有|不包含|无)"),
)


def is_unanswerable(text: str) -> bool:
    """判断模型这条回复是不是「知识库答不了」。

    返回 True 才会触发联网兜底。
    """
    if not text:
        return False
    head = text[:_REFUSAL_SCAN_CHARS]
    return any(pattern.search(head) for pattern in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

MODE_KB = "kb"          # 由知识库回答
MODE_WEB = "web"        # 知识库答不了，联网检索后回答
MODE_REFUSED = "refused"  # 知识库答不了，联网也没成，退回拒答

# 流式输出的扣留字符数。24 个字符足够覆盖「根据现有知识库暂未找到相关信息」，
# 又短到正常回答几乎感觉不到等待。
_HOLDBACK_CHARS = 24


@dataclass
class Answer:
    text: str
    sources: List[RetrievedDoc]
    elapsed_ms: int
    web_sources: List[WebDoc] = field(default_factory=list)
    mode: str = MODE_KB


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

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
        """检索并返回通过相关性阈值的结果（可能为空）。"""
        return self._vs.search(question, top_k or self._top_k)

    # ------------- 提示词与调用 -------------

    def _build_chain(self, context: str, language: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT.format(context=context, language=language)),
                ("human", "{question}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _build_web_chain(self, context: str, language: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    WEB_SYSTEM_PROMPT.format(context=context, language=language),
                ),
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

    # ------------- 联网兜底 -------------

    def web_retrieve(self, question: str) -> List[WebDoc]:
        """联网检索。任何失败都降级为空列表，绝不让兜底层把主流程搞崩。"""
        if not settings.web_fallback_enabled:
            logger.debug("联网兜底已关闭，跳过检索")
            return []
        try:
            return web_search(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网检索失败，退回知识库兜底话术：%s", exc)
            return []

    @staticmethod
    def _kb_source_payload(docs: List[RetrievedDoc]) -> List[dict]:
        return [
            {"file": d.source, "snippet": d.content[:160], "score": d.score}
            for d in docs
        ]

    @staticmethod
    def _web_source_payload(docs: List[WebDoc]) -> List[dict]:
        return [
            {
                "file": d.label,
                "title": d.title,
                "url": d.link,
                "snippet": d.content[:160],
                "score": 0.0,
                "kind": "web",
                "publish_date": d.publish_date,
            }
            for d in docs
        ]

    # ------------- 同步问答 -------------

    def ask(
        self,
        question: str,
        language: str = "中文",
        top_k: Optional[int] = None,
    ) -> Answer:
        start = time.time()
        docs = self.retrieve(question, top_k)

        kb_text = FALLBACK_ANSWER
        if docs:
            chain = self._build_chain(build_context(docs), language)
            kb_text = self._invoke_with_retry(chain, question)
            if not is_unanswerable(kb_text):
                return Answer(
                    text=kb_text,
                    sources=docs,
                    elapsed_ms=int((time.time() - start) * 1000),
                    mode=MODE_KB,
                )
            logger.info("知识库未能回答，转入联网兜底：%s", question)

        web_docs = self.web_retrieve(question)
        if not web_docs:
            return Answer(
                text=kb_text,
                sources=docs,
                elapsed_ms=int((time.time() - start) * 1000),
                mode=MODE_REFUSED,
            )

        chain = self._build_web_chain(build_web_context(web_docs), language)
        try:
            text = self._invoke_with_retry(chain, question)
        except LLMInvocationError as exc:
            logger.warning("联网兜底生成失败，退回知识库兜底话术：%s", exc)
            return Answer(
                text=kb_text,
                sources=docs,
                elapsed_ms=int((time.time() - start) * 1000),
                mode=MODE_REFUSED,
            )

        return Answer(
            text=WEB_DISCLAIMER + text,
            sources=[],
            elapsed_ms=int((time.time() - start) * 1000),
            web_sources=web_docs,
            mode=MODE_WEB,
        )

    def answer_with_web(
        self, question: str, language: str = "中文"
    ) -> Optional[Answer]:
        """跳过知识库，直接联网作答。

        给「知识库已经被别处（ReAct Agent）判定为答不了」的场景用：
        此时再让本流水线重新检索一遍知识库纯属浪费。联网不可用时返回 None，
        调用方自行决定怎么收尾。
        """
        start = time.time()
        web_docs = self.web_retrieve(question)
        if not web_docs:
            return None

        chain = self._build_web_chain(build_web_context(web_docs), language)
        try:
            text = self._invoke_with_retry(chain, question)
        except LLMInvocationError as exc:
            logger.warning("联网兜底生成失败：%s", exc)
            return None

        return Answer(
            text=WEB_DISCLAIMER + text,
            sources=[],
            elapsed_ms=int((time.time() - start) * 1000),
            web_sources=web_docs,
            mode=MODE_WEB,
        )

    # ------------- 流式问答 -------------

    def ask_stream(
        self,
        question: str,
        language: str = "中文",
        top_k: Optional[int] = None,
    ) -> Iterator[dict]:
        """流式问答：依次产出 sources / delta / web_sources / done 事件。

        联网兜底时额外产出 status 事件（告知用户正在联网），
        以及在「已经吐了一部分正文才判定为拒答」时产出 reset 事件（要求前端清空重来）。
        """
        start = time.time()
        docs = self.retrieve(question, top_k)
        yield {"type": "sources", "sources": self._kb_source_payload(docs)}

        kb_text = ""
        if docs:
            kb_text = yield from self._stream_kb(question, language, docs)
        else:
            kb_text = FALLBACK_ANSWER

        if kb_text and not is_unanswerable(kb_text):
            yield {
                "type": "done",
                "elapsed_ms": int((time.time() - start) * 1000),
                "mode": MODE_KB,
            }
            return

        yield from self._stream_web(question, language, docs, kb_text, start)

    def _stream_kb(
        self, question: str, language: str, docs: List[RetrievedDoc]
    ) -> Iterator[dict]:
        """流式产出知识库回答，**返回**累积全文。

        扣留窗口：前 `_HOLDBACK_CHARS` 个字符先不下发。如果这段时间里就能看出
        这是拒答话术，就一个字都不发，直接转联网——用户不会看到「暂未找到」闪一下
        再消失。正常回答只多等一个 chunk 的延迟（几十毫秒）。
        """
        chain = self._build_chain(build_context(docs), language)
        text = ""
        holdback = ""
        emitting = False

        try:
            for chunk in chain.stream({"question": question}):
                if not chunk:
                    continue
                text += chunk
                if emitting:
                    yield {"type": "delta", "text": chunk}
                    continue
                holdback += chunk
                if len(holdback) < _HOLDBACK_CHARS:
                    continue
                if is_unanswerable(holdback):
                    return text
                emitting = True
                yield {"type": "delta", "text": holdback}
                holdback = ""
        except Exception as exc:  # noqa: BLE001
            if text:
                logger.warning("流式输出中断：%s", exc)
            else:
                logger.warning("流式调用失败，降级为同步调用：%s", exc)
                text = self._invoke_with_retry(chain, question)
                if is_unanswerable(text):
                    return text
                emitting = True
                holdback = ""
                yield {"type": "delta", "text": text}

        # 扣留窗口没攒满就结束了：整段短回答还压在 holdback 里，需要补发
        if holdback and not emitting:
            if is_unanswerable(holdback):
                return text
            yield {"type": "delta", "text": holdback}

        return text

    def _stream_web(
        self,
        question: str,
        language: str,
        docs: List[RetrievedDoc],
        kb_text: str,
        start: float,
    ) -> Iterator[dict]:
        """知识库答不了之后的收尾：联网检索 → 流式作答 → done。"""
        logger.info("知识库未能回答，转入联网兜底：%s", question)

        web_docs = self.web_retrieve(question)
        if not web_docs:
            # 联网也没辙：把知识库的兜底话术交出去，行为与改造前一致
            yield {"type": "delta", "text": kb_text or FALLBACK_ANSWER}
            yield {
                "type": "done",
                "elapsed_ms": int((time.time() - start) * 1000),
                "mode": MODE_REFUSED,
            }
            return

        # 已经下发过正文才判定为拒答（长拒答越过扣留窗口）→ 要求前端清空
        yield {"type": "reset"}
        yield {
            "type": "web_sources",
            "sources": self._web_source_payload(web_docs),
        }
        yield {"type": "status", "text": f"校内知识库未收录，已联网检索到 {len(web_docs)} 条资料"}

        chain = self._build_web_chain(build_web_context(web_docs), language)
        received = False
        # 免责声明在**第一个**正文块落地之前注入。若生成彻底失败，
        # 就不会出现「网络内容仅供参考」紧接着一句「暂未找到相关信息」的自相矛盾。
        disclaimer_sent = False
        try:
            for chunk in chain.stream({"question": question}):
                if chunk:
                    received = True
                    if not disclaimer_sent:
                        disclaimer_sent = True
                        yield {"type": "delta", "text": WEB_DISCLAIMER}
                    yield {"type": "delta", "text": chunk}
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网兜底流式输出失败：%s", exc)

        if not received:
            try:
                text = self._invoke_with_retry(chain, question)
                yield {"type": "delta", "text": WEB_DISCLAIMER + text}
            except LLMInvocationError:
                yield {"type": "delta", "text": FALLBACK_ANSWER}

        yield {
            "type": "done",
            "elapsed_ms": int((time.time() - start) * 1000),
            "mode": MODE_WEB,
            "sources": self._web_source_payload(web_docs),
        }
