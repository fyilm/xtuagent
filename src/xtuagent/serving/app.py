"""FastAPI 应用：REST + SSE 流式问答 + 静态前端。"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
from ..core.exceptions import XtuAgentError
from ..core.logging import setup_logging
from .container import STATE_ERROR, STATE_READY, container
from .schemas import AskRequest, AskResponse, HealthResponse, Source

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("XtuAgent 服务启动，后台预热中……")
    container.warmup_async()
    yield
    logger.info("XtuAgent 服务关闭")


app = FastAPI(
    title="XtuAgent API",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> None:
    if container.state == STATE_ERROR:
        raise HTTPException(status_code=503, detail=f"服务异常：{container.error}")
    if container.state != STATE_READY:
        raise HTTPException(status_code=503, detail="服务正在预热，请稍候")


@app.exception_handler(XtuAgentError)
async def domain_error_handler(request: Request, exc: XtuAgentError) -> JSONResponse:
    """领域异常统一降级为 503 + 可读原因（例如 API Key 未配置），而不是 500。"""
    logger.warning("请求 %s 处理失败：%s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    count = container.vector_store.count if container.vector_store else 0
    return HealthResponse(
        status=container.state,
        error=container.error,
        model=settings.zhipu_model,
        index_count=count,
        index_dir=str(settings.index_dir),
    )


@app.get("/api/stats")
def stats() -> dict:
    payload = {
        "status": container.state,
        "model": settings.zhipu_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.retriever_top_k,
        "score_threshold": settings.retriever_score_threshold,
    }
    if container.vector_store is None:
        return payload
    info = container.vector_store.health()
    payload.update({"count": info["count"], "meta": info["meta"]})
    return payload


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    _require_ready()
    answer = container.pipeline.ask(
        req.question.strip(), req.language, top_k=req.top_k
    )
    return AskResponse(
        answer=answer.text,
        sources=[
            Source(file=s.source, snippet=s.content[:160], score=s.score)
            for s in answer.sources
        ],
        elapsed_ms=answer.elapsed_ms,
    )


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    _require_ready()

    def event_stream():
        try:
            for event in container.pipeline.ask_stream(
                req.question.strip(), req.language, top_k=req.top_k
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式问答失败")
            payload = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent")
def agent_ask(req: AskRequest) -> dict:
    """工具增强问答（ReAct Agent，非流式）。

    与 /api/ask 的区别：由模型自主决定是否调用检索/规定原文工具，
    适合需要多步查找的问题。响应结构统一为 {answer, sources, elapsed_ms}。
    """
    _require_ready()
    from ..agent import get_assistant

    start = time.time()
    result = get_assistant().run(req.question.strip())
    return {
        "answer": result.text,
        "sources": [
            {"file": s.source, "snippet": s.content[:160], "score": s.score}
            for s in result.sources
        ],
        "elapsed_ms": int((time.time() - start) * 1000),
    }


if settings.web_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.web_dir), html=True), name="web")
