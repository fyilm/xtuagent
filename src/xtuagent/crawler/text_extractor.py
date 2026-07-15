import os
import re
import logging
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


class TextExtractor:
    def __init__(self, output_dir: str = "data/texts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\x00", "", text)
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    def extract_pdf(self, filepath: str) -> Optional[str]:
        path = Path(filepath)
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(self._clean_text(text))
                return "\n\n".join(pages)
        except Exception as e:
            logger.warning("pdf extract failed: %s — %s", path, e)
            return None

    def extract_and_save(self, filepath: str) -> Optional[str]:
        path = Path(filepath)
        suffix = path.suffix.lower()
        out_name = path.stem + ".txt"
        out_path = self.output_dir / out_name

        if out_path.exists():
            return str(out_path)

        if suffix == ".pdf":
            text = self.extract_pdf(str(path))
        elif suffix in (".md", ".txt"):
            try:
                text = self._clean_text(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, Exception):
                try:
                    text = self._clean_text(path.read_text(encoding="gbk"))
                except Exception:
                    return None
        elif suffix in (".doc", ".docx"):
            logger.warning("doc/docx not supported directly, skip: %s", path)
            return None
        else:
            return None

        if not text or len(text) < 50:
            return None

        out_path.write_text(text, encoding="utf-8")
        logger.info("extracted: %s -> %s", path, out_path)
        return str(out_path)

    def extract_batch(self, filepaths: list[str]) -> list[str]:
        results = []
        for fp in filepaths:
            result = self.extract_and_save(fp)
            if result:
                results.append(result)
        return results
