"""嵌入服务：进程内单例，线程安全，支持预热与离线加载。"""

import logging
import threading
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """bge 嵌入模型单例服务。

    - 双重检查锁保证多线程下只加载一次模型；
    - `warmup()` 供服务启动时预热，避免首次请求长时间等待；
    - 模型文件离线加载（local_files_only），不依赖网络。
    """

    _singleton_lock = threading.Lock()
    _instance: Optional["EmbeddingService"] = None

    def __init__(self) -> None:
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "EmbeddingService":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info(
                        "加载嵌入模型: %s (%s)",
                        settings.embedding_model,
                        settings.embedding_device,
                    )
                    self._model = SentenceTransformer(
                        settings.embedding_model,
                        device=settings.embedding_device,
                        local_files_only=True,
                    )
                    logger.info("嵌入模型加载完成 (dim=%d)", self.dimension)
        return self._model

    def warmup(self) -> None:
        """预热：触发模型加载。"""
        _ = self.model

    @property
    def dimension(self) -> int:
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()  # 兼容旧版

    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = False):
        return self.model.encode(
            texts,
            batch_size=batch_size or settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )


class LangchainEmbeddingAdapter(Embeddings):
    """把 EmbeddingService 适配为 LangChain Embeddings 接口。"""

    def __init__(self, service: Optional[EmbeddingService] = None) -> None:
        self._service = service or EmbeddingService.instance()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._service.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._service.encode([text])[0].tolist()
