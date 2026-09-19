"""检索辅助：上下文拼装。"""

from typing import List

from .vector_store import RetrievedDoc


def build_context(docs: List[RetrievedDoc]) -> str:
    """把检索结果拼装为 Prompt 上下文，附带来源标注。"""
    parts = []
    for doc in docs:
        parts.append(f"[来源: {doc.source}]\n{doc.content}")
    return "\n\n---\n\n".join(parts)
