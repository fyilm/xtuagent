"""API 请求/响应模型。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """一条参考来源。知识库来源与网络来源共用本模型，靠 `kind` 区分。

    - `kind="kb"`：本地知识库片段，`file` 是校内文件名，`score` 是余弦相似度。
    - `kind="web"`：联网检索结果，`file` 是站点名，`url` 可点击跳转，
      `score` 无意义（恒为 0，前端不展示）。
    """

    file: str
    snippet: str
    score: float = 0.0
    kind: str = "kb"
    url: str = ""
    title: str = ""
    publish_date: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language: str = "中文"
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    elapsed_ms: int = 0
    mode: str = "kb"


class HealthResponse(BaseModel):
    status: str
    error: str = ""
    model: str = ""
    index_count: int = 0
    index_dir: str = ""
