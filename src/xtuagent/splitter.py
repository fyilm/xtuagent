import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import config

logger = logging.getLogger(__name__)


def create_splitter(
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> RecursiveCharacterTextSplitter:
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(documents: list) -> list:
    splitter = create_splitter()
    chunks = splitter.split_documents(documents)
    logger.info("split into %d chunks (size=%d, overlap=%d)", len(chunks), config.chunk_size, config.chunk_overlap)
    return chunks
