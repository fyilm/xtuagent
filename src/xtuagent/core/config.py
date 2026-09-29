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
    # ⚠️ chunk_size / chunk_overlap 的单位是 **token**，不是字符。
    # 必须按 token 计数：bge-base-zh 的 max_seq_length=512，中文约 1 字 1 token，
    # 早先按字符切 800 会让 52% 的块超限被静默截断（详见 ingest/splitter.py）。
    # 取 500 而非 512，是给 tokenizer 自动补的 [CLS]/[SEP] 留 2 个 token 余量。
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 5
    # 相关性阈值（余弦相似度）。**默认 0 表示关闭过滤**，检索始终返回 top_k。
    #
    # 为什么不默认开：BGE 中文模型的余弦分布会被压缩在很窄的高分区间，
    # 且区间位置随语料变化很大。实测某校园语料：语料内问题 top-1 落在
    # 0.67~0.69，语料外问题 top-1 落在 0.63~0.67，两者几乎完全重叠——
    # 此时任何固定阈值要么滤掉全部、要么形同虚设。
    #
    # 因此是否拒答交给两层机制处理：
    #   1) 提示词要求模型在知识库无足够信息时如实拒答（默认生效）；
    #   2) 若你在自己的语料上标定出了可分离的阈值，再用本项显式开启。
    # 标定方法：`uv run python scripts/eval_qa.py --skip-llm` 会打印
    # 语料内/语料外问题的 top-1 分数分布，据此判断是否可分离。
    retriever_score_threshold: float = 0.0
    # 同一来源文件最多保留的片段数，避免上下文被单个长文档占满
    retriever_max_per_source: int = 3

    # 服务
    host: str = "127.0.0.1"
    port: int = 8000

    # 爬虫
    # 深度 2 时，作为二级链接被发现的子站（社科处、统战部、期刊社等）只会
    # 抓到首页就不再展开，导致 30+ 个子站只贡献 1 篇文本。调到 3 才能覆盖。
    crawl_max_depth: int = 3
    crawl_delay: float = 0.3
    crawl_timeout: int = 20
    crawl_max_pages_per_site: int = 60
    # 单次运行的全局页数预算。整站爬取耗时很长，设上限可保证「每次运行都有边界」，
    # 未爬完的部分会随断点状态保存，下次运行继续。
    crawl_max_total_pages: int = 3000

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
