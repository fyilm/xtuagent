"""文档加载：把 data/texts 下的文本/PDF 加载为 LangChain Document。"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..core.config import settings

logger = logging.getLogger(__name__)


def _content_key(text: str) -> str:
    """按「去空白后的正文」生成指纹，用于识别同一页面的重复落盘。"""
    normalized = "".join(text.split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def load_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    path = Path(docs_dir or settings.texts_dir)
    if not path.exists():
        logger.error("目录不存在：%s", path)
        return []

    documents: List[Document] = []
    seen: dict[str, str] = {}  # 内容指纹 -> 首次出现的文件名
    skipped_dup = 0
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
            # 同一页面常因多个 URL 别名被落盘成多份完全相同的文本，
            # 若不去重，会把同一内容重复塞进索引、稀释检索结果。
            if docs:
                key = _content_key("\n".join(d.page_content for d in docs))
                if key in seen:
                    skipped_dup += 1
                    logger.debug("跳过重复文档 %s（与 %s 内容相同）", filepath.name, seen[key])
                    continue
                seen[key] = filepath.name
            documents.extend(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载失败 %s：%s", filepath.name, exc)

    logger.info("共加载 %d 篇文档（去重跳过 %d 篇重复）", len(documents), skipped_dup)
    return documents
