"""文档加载：把 data/texts 下的文本/PDF 加载为 LangChain Document。"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..core.config import settings

logger = logging.getLogger(__name__)


def load_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    path = Path(docs_dir or settings.texts_dir)
    if not path.exists():
        logger.error("目录不存在：%s", path)
        return []

    documents: List[Document] = []
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
            documents.extend(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载失败 %s：%s", filepath.name, exc)

    logger.info("共加载 %d 篇文档", len(documents))
    return documents
