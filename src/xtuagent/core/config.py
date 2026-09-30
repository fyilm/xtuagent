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
    # 默认值选型说明（2026-09 实测，同一出口网络、同一提问口径压测）：
    #   glm-4-flash〔默认·免费〕    4/4 成功，极简问题 0.4~0.8s，RAG 端到端 29.8s
    #   glm-4.5-flash〔免费〕       4/4 成功，但抖动大（3.3~40.5s）
    #   glm-4.7-flash〔免费〕       不可用：429 限流挤爆，3 次重试全失败
    #   glm-4.7-flashx〔付费〕      4/7 成功，失败时精确卡死 60.1s，不可用于生产
    #   glm-5.3-flashx〔付费〕      4/4 成功，极简 2.5~4.4s，RAG 端到端 17.2s（更快）
    # 结论：免费档里 glm-4-flash 最稳最快；若可接受付费，glm-5.3-flashx 是全面更优解。
    #
    # 两个必须知道的坑：
    #   1) `GET /api/paas/v4/models` 只枚举主型号，不含 `-flash`/`-flashx` 子型号，
    #      但子型号可直接调用（按账号权限放开）。判断存在性看错误码：
    #      429=存在但限流、403=无权访问、404=不存在。故本默认值未出现在该列表属正常。
    #   2) ChatZhipuAI 的 HTTP 超时硬编码为 60s（langchain_community/chat_models/zhipuai.py
    #      四处 httpx.Client(timeout=60)），且无可配置字段。超过 60s 的生成必被 ReadTimeout
    #      砍断，只能靠 pipeline._invoke_with_retry 兜底。
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
    # 送入模型前，丢弃正文过短的片段（抽取残渣 + 导航碎片）。
    # 定 60 是扫出来的拐点：实测某站点 45 字左右的导航块（"湘潭大学/校内链接/
    # 校外链接/图书馆/招生网…"）会因为 BGE 相似度被压缩而拿到 0.8703，
    # 压过真正的教授简介（0.8667）排到第一 —— 按长度才滤得掉。
    # 不能再往上调：转专业那条 0.8695 的相关附件块约 90 字，阈值设 100 会误删。
    retriever_min_chunk_chars: int = 60
    # 送入模型前对片段做近似去重，避免同一段样板文字占满 top_k 名额（0 关闭）
    retriever_dedup_threshold: float = 0.9
    # 建库时的近似去重阈值（char 5-gram Jaccard）。取 0.95 有实测依据：
    # 无关的两篇通知仅因共用页面模板就能到 0.90~0.95，真正重复的在 0.95 以上。
    # 设为 0 则只保留「逐字节相同」的精确去重。
    docs_near_dup_threshold: float = 0.95

    # ---- 联网兜底 ----
    # 知识库答不出来时，是否自动去网上检索答案。
    #
    # 为什么要这层：本地语料只有校内站点的 4400+ 篇文本，对「校园网 VPN 怎么用」
    # 这类问题常常只有《校园网简介》而无操作步骤；纯域外问题更是完全没有依据。
    # 直接回一句「暂未找到相关信息」等于把用户堵回去——用户要的是答案，不是免责声明。
    #
    # 触发时机：模型读完知识库片段后判定「答不了」的那一刻，才发起联网检索。
    # 正常能答的问题**不产生任何搜索费用**，也不增加延迟。
    # 兜底本身再失败时，仍然退回原来的拒答话术——不会因为多了这一层而变脆。
    web_fallback_enabled: bool = True
    # 搜索档位：search_std（约 0.01 元/次）｜search_pro（更贵、结果更全）
    web_search_engine: str = "search_std"
    web_search_count: int = 5
    web_search_timeout: int = 20

    # 服务
    host: str = "127.0.0.1"
    port: int = 8000

    # 爬虫
    # 深度 2 时，作为二级链接被发现的子站（社科处、统战部、期刊社等）只会
    # 抓到首页就不再展开，导致 30+ 个子站只贡献 1 篇文本。调到 3 才能覆盖。
    crawl_max_depth: int = 3
    crawl_delay: float = 0.3
    crawl_timeout: int = 20
    # 单站页数上限。注意此前 crawl.py 并未把本项传给 Spider（吃的是 Spider 的
    # 默认值 80），所以这个开关形同虚设——现已接线；14 个学院新域名页面较多，
    # 给到 120 让栏目页能展开，总预算仍由 crawl_max_total_pages 兜住。
    crawl_max_pages_per_site: int = 120
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
