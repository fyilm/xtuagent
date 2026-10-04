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

## 语言

`language` 由提问内容自动判定（`core/lang.py`），作用有且只有三处：
提示词里「用哪种语言回答」、拒答时照抄的那句话、以及服务端注入的话术
（兜底答复 / 联网免责声明 / 联网状态提示，见 `core/messages.py`）。
它对**检索与是否联网的判定没有影响**——答得出来就绝不联网，这条成本底线与语言无关。
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
from ..core.lang import DEFAULT_LANGUAGE
from ..core.messages import LANGUAGES, fallback_answer, web_disclaimer, web_status
from .llm import create_llm
from .retriever import build_context
from .vector_store import RetrievedDoc, VectorStoreService
from .websearch import WebDoc, build_web_context, web_search

logger = logging.getLogger(__name__)

# 中文兜底话术的模块级别名。保留它有两个原因：
# 1. 既有引用（脚本、测试、README）不必改；
# 2. 中文是默认语言，绝大多数调用点用的就是它。
# 真正参与运行的调用一律走 `fallback_answer(language)` / `web_disclaimer(language)`。
FALLBACK_ANSWER = fallback_answer(DEFAULT_LANGUAGE)

SYSTEM_PROMPT = """你是湘潭大学（Xiangtan University）的智能学业助手，专门为学生提供准确的学业信息。

请严格依据下面提供的知识库内容回答问题。回答要求：
1. 只使用知识库中出现的信息。不要依赖你自己的先验知识补充细节（尤其是数字、日期、比例、人名）。
2. 如果知识库中没有足够信息回答该问题，请如实回复"{refusal}"，不要猜测。
3. 回答准确、简洁，必要时分点列出，并在句末用 [编号] 标注信息来源（例如"平均学分绩点低于 1.5 会收到学业警告 [1]"）。
4. 回答使用{language}语言。
5. 知识库原文以中文为主。当提问语言不是中文时，把里面的**事实忠实翻译**成{language}作答即可，
   不要因为"知识库不是提问语言"就认为没有相关信息——那会导致明明有据可依却答不出来。

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
   资料多半是中文，提问语言不是中文时**直接翻译**其中的事实即可，
   不要因为语言不一致就当作没查到。

网络检索资料：
---
{context}
---"""

# 联网回答的免责声明。
#
# **由服务端强制注入，不交给模型**。原因：实测 glm-4-flash 会漏掉提示词里
# "必须先说明来源性质"这条要求，直接开讲天气——而"这条信息到底是不是学校官方口径"
# 恰恰是用户最不能搞错的一件事。凡是不能容忍被跳过的内容，都不该指望模型自觉。
#
# 同理，这句话必须**跟着回答语言走**：英文回答前面顶一句中文免责声明，
# 用户会以为整段内容都被降级成了免责套话。多语言文案见 core/messages.py。

# 兼容既有引用（scripts / tests / README）。运行期一律用 web_disclaimer(language)。
WEB_DISCLAIMER = web_disclaimer(DEFAULT_LANGUAGE)
# ---------------------------------------------------------------------------
# 拒答识别
# ---------------------------------------------------------------------------

# 只在回答**开头**这段范围内判定。正文中偶然出现「未收录」不该被当成整条拒答，
# 例如「学分绩点低于 1.5 会收到学业警告。如果没有找到相关豁免条款，请联系教务处」
# 是一句完整的正常回答，绝不能被误判。
_REFUSAL_SCAN_CHARS = 30

# 非中文回答要更长的窗口：同样一句「没找到信息」，中文 10 来个字就说完了，
# 英文要 60 个字符。窗口给太短，英文拒答压根扫不到（曾因此漏判）。
_FOREIGN_SCAN_CHARS = 200

# 全部锚定在回答开头，且只允许极短的引导语（"很抱歉，"这类）。
# 误判的代价是「把好答案换成网络答案」，比漏判严重得多，所以这里刻意保守。
_REFUSAL_PATTERNS = (
    # 提示词给定的原句开头，模型改写后半句也会保留它
    re.compile(r"^根据现有知识库"),
    re.compile(r"^.{0,10}暂未找到相关"),
    re.compile(r"^.{0,10}(未能|无法|没有|不能)找到相关"),
    re.compile(r"^.{0,10}(暂未|尚未|未)收录"),
    re.compile(r"^.{0,10}知识库(中|里)?.{0,6}(暂未|尚未|未|没有|不包含|无)"),
    # 「我（很抱歉）无法直接回答这个问题」。
    # 实测（2026-10-02）：模型拿一堆无关检索结果收尾时就是这么开头的——
    # 「很抱歉，我无法直接回答"什么是摩尔定律？"这个问题。根据我的知识库，我找到了…」。
    # 这种既没答上、又不去联网的回答是两头不讨好，必须能被认出来。
    # 宾语限定在「回答/解答/给出答案」，避免误伤「我无法确认该信息的准确性，但…」
    # 这类正文中正常出现的表述。
    re.compile(r"^.{0,15}(无法|不能|没法)(直接)?(回答|解答|答复|给出\w{0,4}答案)"),
)

# 外文拒答。两个原则：
# 1. 必须同时出现「否定词」和「被否定的对象」（information / knowledge base…），
#    只靠一个 not 会误伤"They cannot be submitted late"这种正常回答；
# 2. 引导语放宽到 120 字符。实测模型常先写一句前言再说拒答：
#    "The GPA warning threshold is not explicitly mentioned in the provided knowledge base.
#     No relevant information was found in the current knowledge base. …"
#    —— 真正的否定词落在第 82 个字符，窗口给 60 就会整个漏掉
#    （漏判的后果：用户只拿到一句拒答，联网兜底白设计了）。
# 放宽窗口不会误伤正常回答，因为「对象」这一半的要求一直很严：
#   "There is no exam scheduled…"（no exam ≠ no information）、
#   "You cannot find the room…"（find 的宾语不在清单里）都不命中。
_FOREIGN_LEAD = r"^.{0,120}"
# 英语的「做不到」类否定词
_EN_UNABLE = r"(?:cannot|can\s*not|can't|could\s*not|couldn't|unable\s+to)"
_FOREIGN_REFUSAL_PATTERNS = (
    # 英语：「没有…（相关/具体/足够的）信息」
    re.compile(
        _FOREIGN_LEAD
        + r"(?:no|not\s+any|without\s+any)\s+(?:\w+\s+){0,2}(?:information|info)\b",
        re.I,
    ),
    # 英语：「无法提供…（答案/信息）」
    re.compile(
        _FOREIGN_LEAD
        + _EN_UNABLE
        + r"\s+(?:\w+\s+){0,2}?(?:provide|give|offer)\s+(?:an?\s+)?"
        + r"(?:\w+\s+){0,2}?(?:answer|information|response)\b",
        re.I,
    ),
    # 英语：「找不到…信息」。
    # 宾语必须落到 information/answer 这类词上：单看"They cannot find the room"
    # 会误伤正常回答，那正是不该触发联网的句子。
    re.compile(
        _FOREIGN_LEAD
        + _EN_UNABLE
        + r"\s+(?:find|locate|determine)\s+"
        + r"(?:any\s+|the\s+|specific\s+|relevant\s+|sufficient\s+|matching\s+){0,2}?"
        + r"(?:information|info|answer|details|data)\b",
        re.I,
    ),
    # 英语：「知识库（不）包含…」
    re.compile(
        _FOREIGN_LEAD
        + r"knowledge\s*base\b[^.!?\n]{0,60}?\b(?:does\s+not|doesn't|do\s+not|don't)"
        + r"\s+(?:\w+\s+){0,2}(?:contain|include|mention|cover|provide)\b",
        re.I,
    ),
    # 日语：「未找到/没有相应信息」
    re.compile(r"^.{0,40}(?:情報が(?:見つかりません|ありません)|該当する情報が見つかりません)"),
    # 韩语：「找不到/没有信息」
    re.compile(r"^.{0,40}(?:찾을\s*수\s*없|정보가\s*없|해당하는\s*정보를?\s*찾지)"),
    # 俄语：「未找到/没有信息/未能」
    re.compile(r"^.{0,40}(?:не\s+найден|не\s+удалось|отсутству|нет\s+информац|не\s+содержит)", re.I),
)


# 回答开头常被模型加上装饰符号：`**加粗**`、`> 引用`、引号、列表点……
# 不归一化的话，"**根据现有知识库…" 会绕过所有以 `^` 锚定的规则。
_LEADING_NOISE_CHARS = " \t\r\n*#>~-_=·•∙\"'“”‘’「」『』【】《》〈〉〔〕:：,，.。!！?？…"
_LEADING_NOISE = re.compile("^[" + re.escape(_LEADING_NOISE_CHARS) + "]+")


def _strip_leading_noise(text: str) -> str:
    """去掉回答开头的装饰符号。"""
    return _LEADING_NOISE.sub("", text)


def _compact(text: str) -> str:
    """去掉所有空白并统一大小写，用于「照抄句」比对。

    模型的排版自由（`No  relevant\ninformation`）不该影响识别。
    """
    return re.sub(r"\s+", "", text).casefold()


# 每种语言兜底话术的「指纹」：压掉空白后的前 12 个字符。
#
# 这是最精确的一路识别——提示词要求模型原样照抄这句话，
# 于是我们不需要猜它怎么改写，直接比对即可，且天然支持任意语言。
# 12 个字符足够区分（"norelevantin" / "現在のナレッジベースには" / "втекущейбазе"），
# 又短到流式扣留窗口（24 字符）里就能比对出来。
_SIGNATURE_LEN = 12

# 指纹**只在开头**这段范围内找，不整篇搜索。
# 放宽到整篇会引入新的误判：正文里引用一句「若知识库暂未找到相关信息，请联系教务处」
# 是正常回答，不是整条拒答。留 28 个字符的余地，够覆盖 "I'm sorry, but " 这类礼帽。
_SIGNATURE_WINDOW = 40

_REFUSAL_SIGNATURES = tuple(
    dict.fromkeys(
        _compact(fallback_answer(lang))[:_SIGNATURE_LEN] for lang in LANGUAGES
    )
)


def is_unanswerable(text: str) -> bool:
    """判断模型这条回复是不是「知识库答不了」。

    返回 True 才会触发联网兜底。判定分三路，从严到宽：

    1. **照抄句指纹**：与提示词要求模型照抄的那句兜底话术对齐，精确、语言无关；
    2. **中文正则**：沿用原有保守设计，应付模型改写；
    3. **外文正则**：英文/日文/韩文/俄文的常见拒答说法，窗口更长。

    三路都只看回答**开头**——正文里偶然提到「未收录」不算整条拒答。
    """
    if not text:
        return False
    head = _strip_leading_noise(text)

    compact = _compact(head)[:_SIGNATURE_WINDOW]
    if any(sig in compact for sig in _REFUSAL_SIGNATURES):
        return True

    if any(p.search(head[:_REFUSAL_SCAN_CHARS]) for p in _REFUSAL_PATTERNS):
        return True

    return any(p.search(head[:_FOREIGN_SCAN_CHARS]) for p in _FOREIGN_REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

MODE_KB = "kb"          # 由知识库回答
MODE_WEB = "web"        # 知识库答不了，联网检索后回答
MODE_REFUSED = "refused"  # 知识库答不了，联网也没成，退回拒答

# 流式输出的扣留字符数。24 个字符足够覆盖「根据现有知识库暂未找到相关信息」，
# 也够覆盖各语言兜底话术的识别指纹（12 字符），又短到正常回答几乎感觉不到等待。
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
                (
                    "system",
                    SYSTEM_PROMPT.format(
                        context=context,
                        language=language,
                        # 拒答话术也跟着语言走。否则英文回答会被要求照抄一句中文，
                        # 模型要么照抄（中英夹杂），要么自己改写（识别不出拒答）。
                        refusal=fallback_answer(language),
                    ),
                ),
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

        kb_text = fallback_answer(language)
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
            text=web_disclaimer(language) + text,
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
            text=web_disclaimer(language) + text,
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
            kb_text = fallback_answer(language)

        if kb_text and not is_unanswerable(kb_text):
            yield {
                "type": "done",
                "elapsed_ms": int((time.time() - start) * 1000),
                "mode": MODE_KB,
                "language": language,
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
            yield {"type": "delta", "text": kb_text or fallback_answer(language)}
            yield {
                "type": "done",
                "elapsed_ms": int((time.time() - start) * 1000),
                "mode": MODE_REFUSED,
                "language": language,
            }
            return

        # 已经下发过正文才判定为拒答（长拒答越过扣留窗口）→ 要求前端清空
        yield {"type": "reset"}
        yield {
            "type": "web_sources",
            "sources": self._web_source_payload(web_docs),
        }
        yield {
            "type": "status",
            "text": web_status(language, len(web_docs)),
        }

        chain = self._build_web_chain(build_web_context(web_docs), language)
        disclaimer = web_disclaimer(language)
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
                        yield {"type": "delta", "text": disclaimer}
                    yield {"type": "delta", "text": chunk}
        except Exception as exc:  # noqa: BLE001
            logger.warning("联网兜底流式输出失败：%s", exc)

        if not received:
            try:
                text = self._invoke_with_retry(chain, question)
                yield {"type": "delta", "text": disclaimer + text}
            except LLMInvocationError:
                yield {"type": "delta", "text": fallback_answer(language)}

        yield {
            "type": "done",
            "elapsed_ms": int((time.time() - start) * 1000),
            "mode": MODE_WEB,
            "language": language,
            "sources": self._web_source_payload(web_docs),
        }
