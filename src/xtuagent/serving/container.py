"""服务容器：持有模型/索引/流水线单例，支持后台预热。"""

import logging
import threading
from typing import Optional

from ..rag.pipeline import RAGPipeline
from ..rag.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_ERROR = "error"


class ServiceContainer:
    """进程级服务容器。

    状态机：idle → loading → ready / error
    `/health` 据此向客户端反馈就绪情况。
    """

    def __init__(self) -> None:
        self.state: str = STATE_IDLE
        self.error: str = ""
        self.vector_store: Optional[VectorStoreService] = None
        self.pipeline: Optional[RAGPipeline] = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        """同步预热：加载嵌入模型 + 索引 + 构建流水线。"""
        with self._lock:
            if self.state == STATE_READY:
                return
            self.state = STATE_LOADING
            self.error = ""

        try:
            logger.info("服务预热开始……")
            from ..rag.embeddings import EmbeddingService
            from ..rag.vector_store import load_index

            EmbeddingService.instance().warmup()
            self.vector_store = load_index()
            self.pipeline = RAGPipeline(self.vector_store)
            self.state = STATE_READY
            logger.info("服务预热完成，索引条目：%d", self.vector_store.count)
        except Exception as exc:  # noqa: BLE001
            self.state = STATE_ERROR
            self.error = str(exc)
            logger.exception("服务预热失败")

    def warmup_async(self) -> None:
        """后台线程预热，不阻塞服务启动。"""
        thread = threading.Thread(target=self.warmup, name="warmup", daemon=True)
        thread.start()


container = ServiceContainer()
