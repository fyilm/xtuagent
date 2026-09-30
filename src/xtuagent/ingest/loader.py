"""文档加载：把 data/texts 下的文本/PDF 加载为 LangChain Document。

去重分两级：
1. **精确去重**：去空白后逐字节相同（同一页面的多个 URL 别名）；
2. **近似去重**：char 5-gram Jaccard ≥ 阈值（默认 0.95）——覆盖「同一篇稿件
   被发到多个栏目、路径/栏目号不同，正文基本一样、只差页尾」这类情况，
   精确去重抓不到。

阈值取 0.95 而不是 0.85/0.90 是有实测依据的：校园站点大量共用页面模板，
无关的两篇通知（如「普通话测试」vs「四六级报名」）仅因共用顶栏/页脚就能达到
0.90–0.95，真正重复的是 0.95 以上那一档。阈值调低会成片误删。
"""

import hashlib
import logging
import zlib
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..core.config import settings

logger = logging.getLogger(__name__)

# 近似去重的参数
_SHINGLE = 5          # 字符 n-gram 长度
_SKETCH = 64          # 每篇文档保留的最小哈希数（bottom-k 草图）
_MIN_CANDIDATE = 20   # 草图交集至少这么多才做精确 Jaccard，避免误配
_MIN_CHARS = 200      # 归一化后短于此长度不做近似去重（太短的重合没意义）


def _content_key(text: str) -> str:
    """按「去空白后的正文」生成指纹，用于识别同一页面的重复落盘。"""
    normalized = "".join(text.split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def _sketch(text: str, limit: int = 3000):
    """返回 (归一化正文, 5-gram 集合, bottom-k 草图)。

    用 crc32 而非内置 hash()：内置 hash 对 str 每个进程加随机盐，
    会导致同一批语料在不同次构建里删掉不同的文档。
    """
    normalized = "".join(text.split())[:limit]
    if len(normalized) < _MIN_CHARS:
        return normalized, None, None
    grams = {zlib.crc32(normalized[i : i + _SHINGLE].encode()) for i in range(len(normalized) - _SHINGLE + 1)}
    return normalized, grams, frozenset(sorted(grams)[:_SKETCH])


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class _NearDupFilter:
    """流式近似去重：维护草图倒排索引，候选对再做精确 Jaccard。"""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._index: dict[int, list[int]] = {}
        self._kept: list[tuple[set, str]] = []  # (grams, 文件名)

    def is_duplicate(self, text: str, name: str) -> Optional[str]:
        """重复则返回「与谁重复」的文件名，否则登记并返回 None。"""
        _, grams, sketch = _sketch(text)
        if grams is None:
            return None

        hits: dict[int, int] = {}
        for h in sketch:
            for idx in self._index.get(h, ()):
                hits[idx] = hits.get(idx, 0) + 1

        for idx, shared in hits.items():
            if shared < _MIN_CANDIDATE:
                continue
            other_grams, other_name = self._kept[idx]
            if _jaccard(grams, other_grams) >= self.threshold:
                return other_name

        idx = len(self._kept)
        self._kept.append((grams, name))
        for h in sketch:
            self._index.setdefault(h, []).append(idx)
        return None


def load_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    path = Path(docs_dir or settings.texts_dir)
    if not path.exists():
        logger.error("目录不存在：%s", path)
        return []

    threshold = getattr(settings, "docs_near_dup_threshold", 0.95)
    near_filter = _NearDupFilter(threshold) if threshold and threshold > 0 else None

    documents: List[Document] = []
    seen: dict[str, str] = {}  # 内容指纹 -> 首次出现的文件名
    skipped_dup = 0
    skipped_near = 0
    for filepath in sorted(path.iterdir()):
        if not filepath.is_file():
            continue
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(str(filepath)).load()
            elif suffix in (".txt", ".md"):
                docs = TextLoader(str(filepath), encoding="utf-8").load()
            else:
                continue
            for doc in docs:
                doc.metadata["source_file"] = filepath.name

            joined = "\n".join(d.page_content for d in docs)

            # 一级：同一页面因多个 URL 别名被落盘成多份完全相同的文本。
            if docs:
                key = _content_key(joined)
                if key in seen:
                    skipped_dup += 1
                    logger.debug("跳过重复文档 %s（与 %s 内容相同）", filepath.name, seen[key])
                    continue
                seen[key] = filepath.name

            # 二级：正文基本一致、只差页尾/栏目信息（精确指纹不同，上面抓不到）。
            if near_filter is not None and docs:
                dup_of = near_filter.is_duplicate(joined, filepath.name)
                if dup_of is not None:
                    skipped_near += 1
                    logger.debug(
                        "跳过近似重复文档 %s（与 %s 相似度 ≥ %.2f）",
                        filepath.name,
                        dup_of,
                        threshold,
                    )
                    continue

            documents.extend(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载失败 %s：%s", filepath.name, exc)

    logger.info(
        "共加载 %d 篇文档（精确去重跳过 %d 篇，近似去重跳过 %d 篇，阈值 %.2f）",
        len(documents),
        skipped_dup,
        skipped_near,
        threshold,
    )
    return documents
