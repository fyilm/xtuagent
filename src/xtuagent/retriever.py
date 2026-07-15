import logging
from typing import Optional

from langchain_community.vectorstores import Chroma

from .config import config

logger = logging.getLogger(__name__)


def create_retriever(vector_store: Chroma, top_k: Optional[int] = None):
    top_k = top_k or config.retriever_top_k
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def search_similar(
    vector_store: Chroma,
    query: str,
    top_k: Optional[int] = None,
) -> list:
    top_k = top_k or config.retriever_top_k
    docs = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    results = [doc for doc, score in docs]
    logger.debug("retrieved %d docs for query: %s", len(results), query[:50])
    return results
