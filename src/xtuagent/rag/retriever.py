"""检索辅助：上下文拼装。"""

from typing import List

from .vector_store import RetrievedDoc


def build_context(docs: List[RetrievedDoc]) -> str:
    """把检索结果拼装为 Prompt 上下文。

    每个片段带编号与来源，便于模型在回答中标注引用（如「[1]」）。

    这里**不把相似度写进上下文**：BGE 中文模型的余弦分数在语料内部区分度有限
    （相关/无关常落在同一区间），把「相关度 0.67」这类数字交给模型，
    反而可能诱导它把不相关内容当成依据。相关性判断应交给模型读内容本身。
    """
    parts = []
    for idx, doc in enumerate(docs, 1):
        parts.append(f"[{idx}] 来源：{doc.source}\n{doc.content}")
    return "\n\n---\n\n".join(parts)
