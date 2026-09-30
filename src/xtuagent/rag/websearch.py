"""联网兜底检索：知识库答不出来时，去网上把答案找回来。

## 为什么需要这一层

本地语料只覆盖校内站点（4400+ 篇文本、80+ 个来源）。对「校园网 VPN 怎么用」
这类问题，知识库往往只有《校园网简介》而**没有操作步骤**；对纯域外问题
（天气、常识、代码）则完全没有依据。此前的做法是丢一句
「根据现有知识库暂未找到相关信息」，把问题原样退还给用户——这不是回答，
这是把服务窗口关了。

## 为什么用独立搜索 API，而不是对话接口内置的 web_search 工具

内置工具把「搜什么、搜几条、怎么综合」全部交给模型，引用格式不可控，
也无法复用本项目已有的提示词与来源展示链路。独立 API 拿回的是**结构化结果**
（title / link / media / content / publish_date），可以像知识库片段一样统一编号、
引用、在「参考来源」里展开并点击跳转；而且**只在知识库真答不出来时才付费**。

## 计费

`search_std` 每次调用约 0.01 元，且仅在拒答后触发，正常问答不产生费用。
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import httpx

from ..core.config import settings
from ..core.exceptions import WebSearchError
from .llm import validate_api_key

logger = logging.getLogger(__name__)

WEB_SEARCH_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"

# 单条网页正文截断长度。搜索结果正文动辄两三千字，5 条就能把上下文塞爆，
# 且会把真正有用的操作步骤淹没在官网简介里——截到 800 字足够承载步骤型信息。
CONTENT_LIMIT = 800


@dataclass
class WebDoc:
    """一条网络检索结果。

    字段对齐智谱搜索 API：`media` 是站点名（如「湘潭大学」），
    `publish_date` 可能是空串（很多校园页面没有标注日期）。
    """

    title: str
    link: str
    media: str
    content: str
    publish_date: str = ""
    # 网络结果没有余弦相似度，恒为 0；前端据 kind 字段区分展示，不读这个值。
    score: float = 0.0

    @property
    def label(self) -> str:
        """展示用的来源名：站点名优先，缺失时退回域名。"""
        if self.media:
            return self.media
        if self.link:
            return self.link.split("/")[2] if "//" in self.link else self.link
        return "网络来源"


def web_search(
    query: str,
    count: Optional[int] = None,
    engine: Optional[str] = None,
    timeout: Optional[int] = None,
) -> List[WebDoc]:
    """联网检索。失败抛 WebSearchError，由调用方决定是否降级。

    这里刻意不吞异常：调用方需要区分「搜到了但为空」和「根本没搜成」，
    前者可以如实告知用户网上也没有，后者只能退回知识库的兜底话术。
    """
    query = (query or "").strip()
    if not query:
        return []

    api_key = validate_api_key()
    limit = count or settings.web_search_count
    payload = {
        "search_engine": engine or settings.web_search_engine,
        "search_query": query,
        "count": limit,
    }

    try:
        response = httpx.post(
            WEB_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout or settings.web_search_timeout,
        )
    except httpx.HTTPError as exc:
        raise WebSearchError(f"联网检索请求失败：{exc}") from exc

    if response.status_code != 200:
        raise WebSearchError(
            f"联网检索返回 {response.status_code}：{response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise WebSearchError("联网检索返回的不是合法 JSON") from exc

    items = data.get("search_result") or []
    docs: List[WebDoc] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        docs.append(
            WebDoc(
                title=(item.get("title") or "").strip(),
                link=(item.get("link") or "").strip(),
                media=(item.get("media") or "").strip(),
                content=content[:CONTENT_LIMIT],
                publish_date=(item.get("publish_date") or "").strip(),
            )
        )

    logger.info("联网检索「%s」命中 %d 条", query, len(docs))
    # 实测该 API 对模糊问题会做多意图拆解（`search_intent` 里能看到），
    # 于是 count=5 也可能回 10 条。这里按配置硬截断，保证上下文长度可控——
    # 10 条 × 800 字已经能把模型淹掉了。
    return docs[:limit]


def build_web_context(docs: List[WebDoc]) -> str:
    """把网络结果拼成提示词上下文，编号与知识库片段保持一致（[1]、[2]…）。"""
    parts = []
    for idx, doc in enumerate(docs, 1):
        head = f"[{idx}] 标题：{doc.title or '（无标题）'}\n站点：{doc.label}"
        if doc.publish_date:
            head += f"\n发布时间：{doc.publish_date}"
        if doc.link:
            head += f"\n链接：{doc.link}"
        parts.append(f"{head}\n正文：{doc.content}")
    return "\n\n---\n\n".join(parts)
