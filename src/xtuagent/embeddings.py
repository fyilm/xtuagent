import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from .config import config

logger = logging.getLogger(__name__)


class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str = "", model: str = "embedding-2"):
        self.api_key = api_key or config.zhipu_api_key
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._ensure_client()
        result = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for attempt in range(5):
                try:
                    resp = self._client.embeddings.create(model=self.model, input=batch)
                    result.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    wait = 3 ** attempt
                    logger.warning("embedding batch %d failed (attempt %d/5), retry in %ds",
                                   i // batch_size, attempt + 1, wait)
                    import time
                    time.sleep(wait)
            import time
            time.sleep(2.0)
        return result

    def embed_query(self, text: str) -> List[float]:
        self._ensure_client()
        for attempt in range(5):
            try:
                resp = self._client.embeddings.create(model=self.model, input=text)
                return resp.data[0].embedding
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 3 ** attempt
                logger.warning("embed_query failed (attempt %d/5), retry in %ds", attempt + 1, wait)
                import time
                time.sleep(wait)
        raise RuntimeError("embed query failed after 5 attempts")


def create_embeddings():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("loading embedding model: %s on %s (offline mode)",
                config.embedding_model, config.embedding_device)
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        multi_process=False,
    )
