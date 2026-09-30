"""FAISS 向量库服务：构建、加载、检索、元数据校验。"""

import contextlib
import hashlib
import json
import logging
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..core.config import settings
from ..core.exceptions import IndexMetaMismatchError, IndexNotReadyError
from .embeddings import LangchainEmbeddingAdapter

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "index_meta.json"


def _jaccard(a: set, b: set) -> float:
    """两个集合的 Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass
class RetrievedDoc:
    """检索结果。score 为 [0,1] 的余弦相似度，越大越相关。"""

    content: str
    source: str
    score: float


@dataclass
class IndexMeta:
    created_at: str = ""
    embedding_model: str = ""
    dimension: int = 0
    vector_count: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    documents: int = 0
    app_version: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "IndexMeta":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class VectorStoreService:
    """FAISS 索引的统一访问入口。

    索引使用内积等价距离（L2）度量，但文档向量与查询向量都做了 L2 归一化，
    因此 ``cos = 1 - distance / 2`` 可以直接还原为真实余弦相似度，
    使对外暴露的 score 具有明确、可解释的语义（而非单调但无意义的变换）。
    """

    def __init__(
        self,
        index: FAISS,
        meta: Optional[IndexMeta] = None,
        embedding_fn=None,
    ) -> None:
        self._index = index
        self.meta = meta
        # 显式注入编码器；兜底兼容 langchain-community 不同版本的属性命名
        self._embedding = (
            embedding_fn
            or getattr(index, "embedding", None)
            or getattr(index, "embeddings", None)
        )

    @property
    def count(self) -> int:
        return int(self._index.index.ntotal)

    def _query_vector(self, query: str) -> np.ndarray:
        """把查询编码为 L2 归一化向量，保证与索引内向量同尺度。"""
        raw = np.asarray(self._embedding.embed_query(query), dtype="float32")
        norm = float(np.linalg.norm(raw))
        return raw / norm if norm > 0 else raw

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        max_per_source: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
        dedup_threshold: Optional[float] = None,
    ) -> List[RetrievedDoc]:
        """检索并按需过滤。

        - `min_score`：相关性阈值。**默认取配置值，而配置值默认为 0 即不过滤**，
          检索会稳定返回 top_k 条；仅当你在自己语料上标定出可分离的阈值后才建议开启。
        - `max_per_source`：同一来源最多保留的片段数，避免上下文被单个长文档占满。
        - `min_chunk_chars`：丢弃正文过短的片段（默认取配置值，0 关闭）。语料里存在
          「。」「..」「.7亿元」这类抽取残渣，它们没有信息量却会占掉一个 top_k 名额。
        - `dedup_threshold`：对返回的片段做近似去重，默认取配置值（0 关闭）。
          校园站点同一段顶栏/页脚会出现在几十个页面上，不去重会出现「top_k 全被
          同一段样板文字占满」；注意按来源去重的 `max_per_source` 挡不住这种跨文件同文。

        过滤后可能返回空列表，此时调用方应给出兜底回答。
        """
        k = top_k or settings.retriever_top_k
        threshold = (
            settings.retriever_score_threshold if min_score is None else min_score
        )
        limit = (
            settings.retriever_max_per_source if max_per_source is None else max_per_source
        )
        min_chars = (
            getattr(settings, "retriever_min_chunk_chars", 0)
            if min_chunk_chars is None
            else min_chunk_chars
        ) or 0
        dedup_th = (
            getattr(settings, "retriever_dedup_threshold", 0.0)
            if dedup_threshold is None
            else dedup_threshold
        ) or 0.0
        filtering = threshold > 0.0
        # 过短片段过滤和近似去重都会消耗名额，所以多取候选，最后再截到 k。
        # 否则「同一段样板文字出现 5 次」会把 top_k 占满，返回的却是 1 条内容；
        # 过滤偏严时还可能凑不满 top_k，留 5 倍余量。
        fetch_k = k * 5 if (min_chars > 0 or dedup_th > 0.0) else k

        pairs = self._index.similarity_search_with_score_by_vector(
            self._query_vector(query), k=fetch_k
        )

        results: List[RetrievedDoc] = []
        per_source: Dict[str, int] = {}
        dropped = 0
        dropped_short = 0
        dropped_dup = 0
        accepted_grams: List[set] = []

        for doc, distance in pairs:
            if len(results) >= k:
                break
            score = 1.0 - float(distance) / 2.0
            score = min(1.0, max(0.0, score))
            if filtering and score < threshold:
                dropped += 1
                continue

            flat = "".join(doc.page_content.split())
            # 纯「。」「..」「.7亿元」这类碎块没有信息量，白占上下文
            if min_chars > 0 and len(flat) < min_chars:
                dropped_short += 1
                continue

            grams = None
            if dedup_th > 0.0:
                # 不要在这里加「长度够长才去重」的门槛：最需要去重的恰恰是短块——
                # 实测某站点 23 个页面尾部都是同一段 43 字符的地址行，
                # 长这样的小块两两 Jaccard = 1.000，却会因为「太短」被漏掉。
                window = flat[:3000]
                grams = {
                    zlib.crc32(window[i : i + 5].encode())
                    for i in range(len(window) - 4)
                }
                if any(_jaccard(grams, g) >= dedup_th for g in accepted_grams):
                    dropped_dup += 1
                    continue

            source = str(
                doc.metadata.get("source_file")
                or doc.metadata.get("source")
                or "unknown"
            )
            if per_source.get(source, 0) >= limit:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            if grams is not None:
                accepted_grams.append(grams)

            results.append(
                RetrievedDoc(
                    content=doc.page_content,
                    source=source,
                    score=round(score, 4),
                )
            )

        if dropped or dropped_short or dropped_dup:
            logger.info(
                "检索「%s」：返回 %d 条（低相关丢弃 %d，过短丢弃 %d，近似重复丢弃 %d）",
                query,
                len(results),
                dropped,
                dropped_short,
                dropped_dup,
            )
        return results

    def health(self) -> dict:
        return {
            "count": self.count,
            "dimension": int(self._index.index.d),
            "meta": self.meta.to_dict() if self.meta else None,
        }


def compute_corpus_fingerprint(texts: List[str]) -> str:
    """计算语料指纹：对全部文本内容做哈希。

    断点续传必须能识别「语料变了」这件事。早期版本只取前 3 条 + 总数，
    语料中间发生替换时指纹不变，会导致旧嵌入被复用到新文本上（静默索引损坏）。
    这里改为全量内容哈希，代价是 O(总字符数) 的纯 CPU 计算，可忽略。
    """
    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(str(len(texts)).encode("ascii"))
    return digest.hexdigest()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，保证检索侧 cos = 1 - L2²/2 成立。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_index(
    chunks: List[Document],
    index_dir: Optional[Path] = None,
    documents: int = 0,
    embedding_fn=None,
    checkpoint_batch: int = 128,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> VectorStoreService:
    """构建并持久化 FAISS 索引（支持断点续传）。

    嵌入计算在 CPU 上耗时较长，进度每 checkpoint_batch 个文本块
    落盘一次，中断后重新运行可自动续算。
    """
    index_dir = Path(index_dir or settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedding_fn = embedding_fn or LangchainEmbeddingAdapter()

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    total = len(texts)
    logger.info("构建 FAISS 索引：%d 个文本块", total)

    fingerprint = compute_corpus_fingerprint(texts)
    progress_file = index_dir / ".build_progress.npz"

    embeddings: List[List[float]] = []
    start = 0
    if progress_file.exists():
        try:
            data = np.load(progress_file, allow_pickle=False)
            if str(data["fingerprint"]) == fingerprint and int(data["done"]) <= total:
                embeddings = data["embeddings"].tolist()
                start = int(data["done"])
                logger.info("检测到构建进度，从第 %d/%d 块继续", start, total)
            else:
                logger.info("语料已变更，忽略旧进度并重新构建索引")
        except Exception as exc:  # noqa: BLE001
            logger.warning("进度文件损坏，重新开始：%s", exc)
            embeddings = []
            start = 0

    for i in range(start, total, checkpoint_batch):
        batch_texts = texts[i:i + checkpoint_batch]
        batch_embs = embedding_fn.embed_documents(batch_texts)
        embeddings.extend(batch_embs)
        done = i + len(batch_texts)
        np.savez(
            progress_file,
            embeddings=np.asarray(embeddings, dtype="float32"),
            done=done,
            fingerprint=fingerprint,
        )
        logger.info("嵌入进度：%d/%d（%.0f%%）", done, total, done / total * 100)

    logger.info("全部嵌入完成，组装 FAISS 索引……")
    matrix = _l2_normalize(np.asarray(embeddings, dtype="float32"))
    index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, matrix.tolist())),
        embedding=embedding_fn,
        metadatas=metadatas,
    )
    index.save_local(str(index_dir))

    meta = IndexMeta(
        created_at=datetime.now().isoformat(timespec="seconds"),
        embedding_model=settings.embedding_model,
        dimension=int(index.index.d),
        vector_count=int(index.index.ntotal),
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=(
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        ),
        documents=documents,
        app_version=settings.app_version,
    )
    (index_dir / META_FILE).write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 索引与元数据都已落盘，进度文件只是缓存，清理失败不应中断构建
    # （沙箱/安全删除钩子可能把 unlink 劫持为回收站操作并使其中止）。
    with contextlib.suppress(OSError):
        if progress_file.exists():
            progress_file.unlink()

    logger.info("索引已保存：%s（%d 条向量）", index_dir, meta.vector_count)
    return VectorStoreService(index, meta, embedding_fn=embedding_fn)


def load_index(
    index_dir: Optional[Path] = None,
    embedding_fn=None,
) -> VectorStoreService:
    """加载 FAISS 索引，并校验元数据。"""
    index_dir = Path(index_dir or settings.index_dir)
    faiss_file = index_dir / INDEX_FILE
    if not faiss_file.exists():
        raise IndexNotReadyError(f"索引不存在：{faiss_file}，请先运行构建索引")

    meta: Optional[IndexMeta] = None
    meta_file = index_dir / META_FILE
    if meta_file.exists():
        meta = IndexMeta.from_dict(json.loads(meta_file.read_text(encoding="utf-8")))
        if meta.embedding_model and meta.embedding_model != settings.embedding_model:
            raise IndexMetaMismatchError(
                f"索引由「{meta.embedding_model}」构建，当前配置为「{settings.embedding_model}」，请重建索引"
            )

    embedding_fn = embedding_fn or LangchainEmbeddingAdapter()
    index = FAISS.load_local(
        str(index_dir), embedding_fn, allow_dangerous_deserialization=True
    )
    service = VectorStoreService(index, meta, embedding_fn=embedding_fn)
    logger.info("索引加载完成：%d 条向量（%s）", service.count, index_dir)
    return service
