"""API 请求/响应模型。"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    file: str
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language: str = "中文"
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    elapsed_ms: int = 0


class HealthResponse(BaseModel):
    status: str
    error: str = ""
    model: str = ""
    index_count: int = 0
    index_dir: str = ""
