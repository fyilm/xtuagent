"""统一配置中心（pydantic-settings）。

设计要点：
1. 环境变量在模块导入时立即设置（早于任何 ML 库加载）；
2. 所有路径基于项目根目录绝对化，与工作目录无关；
3. 启动时即完成类型校验，配置错误立刻暴露。
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：src/xtuagent/core/config.py -> 上溯 3 级
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 以下环境变量必须在导入 sentence-transformers/torch 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 应用
    app_name: str = "XtuAgent"
    app_version: str = "2.0.0"
    log_level: str = "INFO"

    # LLM（仅智谱）
    zhipu_api_key: str = ""
    zhipu_model: str = "glm-4-flash"
    llm_temperature: float = 0.3
    llm_max_retries: int = 3

    # 嵌入模型
    embedding_model: str = "BAAI/bge-base-zh"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # 路径
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    texts_dir: Path = PROJECT_ROOT / "data" / "texts"
    raw_pdfs_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    logs_dir: Path = PROJECT_ROOT / "logs"
    web_dir: Path = PROJECT_ROOT / "web"

    # RAG 参数
    chunk_size: int = 800
    chunk_overlap: int = 50
    retriever_top_k: int = 5

    # 服务
    host: str = "127.0.0.1"
    port: int = 8000

    # 爬虫
    crawl_max_depth: int = 2
    crawl_delay: float = 0.3
    crawl_timeout: int = 20
    crawl_max_pages_per_site: int = 60

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.index_dir,
            self.texts_dir,
            self.raw_pdfs_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
