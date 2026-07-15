import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

LLMProvider = Literal["zhipu", "qianfan"]


@dataclass
class Config:
    llm_provider: LLMProvider = "zhipu"
    zhipu_api_key: str = field(default_factory=lambda: os.getenv("ZHIPU_API_KEY", ""))
    zhipu_model: str = "glm-4-flash"
    qianfan_api_key: str = field(default_factory=lambda: os.getenv("QIANFAN_ACCESS_KEY", ""))
    qianfan_secret_key: str = field(default_factory=lambda: os.getenv("QIANFAN_SECRET_KEY", ""))
    qianfan_model: str = "ernie-lite-8k"

    embedding_model: str = "BAAI/bge-base-zh"
    embedding_device: str = "cpu"

    chroma_persist_dir: str = field(default_factory=lambda: str(_PROJECT_ROOT / "data" / "chroma_db"))
    chroma_collection: str = "xtu_knowledge"

    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 5

    crawl_max_depth: int = 3
    crawl_delay: float = 1.0
    crawl_timeout: int = 30

    docs_dir: str = "docs"
    raw_pdfs_dir: str = "data/raw_pdfs"
    texts_dir: str = "data/texts"

    default_language: str = "Chinese"
    available_languages: list = field(default_factory=lambda: ["Chinese", "English", "French", "German"])


config = Config()
