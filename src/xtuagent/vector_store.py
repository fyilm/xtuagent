import logging
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import config

logger = logging.getLogger(__name__)

_MODEL: Optional[SentenceTransformer] = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        logger.info("loading embedding model: %s on %s", config.embedding_model, config.embedding_device)
        _MODEL = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
            local_files_only=True,
        )
    return _MODEL


class XtuEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        embs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return emb[0].tolist()


def build_vector_store(chunks: list, persist_dir: Optional[str] = None):
    persist_dir = persist_dir or config.chroma_persist_dir
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    embedding = XtuEmbeddings()
    logger.info("building FAISS index for %d chunks", len(chunks))
    index = FAISS.from_documents(documents=chunks, embedding=embedding)
    index.save_local(str(persist_path))
    logger.info("FAISS index saved: %d vectors at %s", index.index.ntotal, persist_path)
    return index


def load_vector_store(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or config.chroma_persist_dir
    persist_path = Path(persist_dir)
    if not (persist_path / "index.faiss").exists():
        logger.warning("FAISS index not found at %s", persist_path)
        return None

    embedding = XtuEmbeddings()
    index = FAISS.load_local(
        str(persist_path), embedding, allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded: %d vectors from %s", index.index.ntotal, persist_path)
    return index
