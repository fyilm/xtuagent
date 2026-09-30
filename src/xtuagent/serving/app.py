"""FastAPI 应用：REST + SSE 流式问答 + 静态前端。"""

import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import settings
from ..core.exceptions import XtuAgentError
from ..core.logging import setup_logging
from ..rag.pipeline import RAGPipeline
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
        "web_fallback": settings.web_fallback_enabled,
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
    sources = RAGPipeline._kb_source_payload(answer.sources) + (
        RAGPipeline._web_source_payload(answer.web_sources)
    )
    return AskResponse(
        answer=answer.text,
        sources=[Source(**s) for s in sources],
        elapsed_ms=answer.elapsed_ms,
        mode=answer.mode,
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
    sources = RAGPipeline._kb_source_payload(result.sources) + (
        RAGPipeline._web_source_payload(result.web_sources)
    )
    return {
        "answer": result.text,
        "sources": sources,
        "elapsed_ms": int((time.time() - start) * 1000),
        "mode": "web" if result.web_sources else "kb",
    }


class NoCacheStaticFiles(StaticFiles):
    """静态前端不做强缓存。

    背景：StaticFiles 默认只回 etag / last-modified，浏览器会走启发式缓存。
    曾出现「改了 web/js/app.js，代码确实是新的，但页面仍执行旧版」——
    表现为改完前端毫无效果，且**需要用户自己去清缓存**，这是不可接受的设计。
    这里统一加 no-cache, must-revalidate：每次都会向服务端校验，
    未变更时走 304，开销可忽略；变更后普通刷新即生效。
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


def static_build_id() -> str:
    """按前端资源内容算出的构建号（前 10 位）。

    作用：把 index.html 里的 `?v={{BUILD}}` 换成它。
    文件内容一变，构建号就变，资源 URL 随之改变——浏览器眼里这是「新文件」，
    必然重新拉取。**不需要人工维护版本号，重启即上新。**
    """
    digest = hashlib.sha1()
    for pattern in ("js/*.js", "css/*.css"):
        for path in sorted(settings.web_dir.glob(pattern)):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


@app.get("/", include_in_schema=False)
def index_page() -> HTMLResponse:
    """首页：注入构建号，避免用户被浏览器缓存困住。"""
    html = (settings.web_dir / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        html.replace("{{BUILD}}", static_build_id()),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/api/build", include_in_schema=False)
def build_info() -> dict:
    """当前前端构建号，便于排查「页面跑的是哪一版」。"""
    return {"build": static_build_id()}


if settings.web_dir.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(settings.web_dir), html=True), name="web")
