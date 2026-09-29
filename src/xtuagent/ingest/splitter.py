"""语义分块：RecursiveCharacterTextSplitter 中文优化。"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings

logger = logging.getLogger(__name__)

# 中文优先的分隔符顺序：段落 → 换行 → 中文句读 → 西文句读 → 空格 → 字符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: list,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list:
    """分块。参数显式传入，不修改全局配置，便于同进程内做多组参数对比实验。"""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = create_splitter(size, overlap)
    chunks = splitter.split_documents(documents)
    logger.info("分块完成：%d 个文本块（大小 %d / 重叠 %d）", len(chunks), size, overlap)
    return chunks
