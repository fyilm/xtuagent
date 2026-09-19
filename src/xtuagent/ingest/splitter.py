"""语义分块：RecursiveCharacterTextSplitter 中文优化。"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings

logger = logging.getLogger(__name__)


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(documents: list) -> list:
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)
    logger.info("分块完成：%d 个文本块（大小 %d / 重叠 %d）", len(chunks), settings.chunk_size, settings.chunk_overlap)
    return chunks
