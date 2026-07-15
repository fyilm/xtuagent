import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)


def load_documents(docs_dir: str = "data/texts") -> list:
    path = Path(docs_dir)
    if not path.exists():
        logger.error("directory not found: %s", path)
        return []

    documents = []
    for filepath in path.iterdir():
        if not filepath.is_file():
            continue
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
            elif suffix in (".txt", ".md"):
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs = loader.load()
            else:
                continue
            for doc in docs:
                doc.metadata["source_file"] = filepath.name
            documents.extend(docs)
            logger.info("loaded: %s (%d chunks)", filepath.name, len(docs))
        except Exception as e:
            logger.warning("failed to load %s: %s", filepath.name, e)

    logger.info("total loaded documents: %d", len(documents))
    return documents
