import os
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class Downloader:
    def __init__(self, output_dir: str = "data/raw_pdfs", timeout: int = 60, workers: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.workers = workers
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def _derive_filename(self, url: str) -> str:
        parsed = urlparse(url)
        path_part = parsed.path.strip("/")
        if not path_part:
            host_part = parsed.hostname or "unknown"
            return f"{host_part}_{abs(hash(url))}.pdf"
        name = path_part.replace("/", "_")
        if any(name.lower().endswith(ext) for ext in (".pdf", ".doc", ".docx", ".md", ".txt")):
            return name
        return f"{name}.pdf"

    def download_one(self, url: str) -> Optional[str]:
        filename = self._derive_filename(url)
        filepath = self.output_dir / filename
        if filepath.exists():
            logger.info("skip existing: %s", filepath)
            return str(filepath)

        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True, stream=True)
            if resp.status_code != 200:
                logger.warning("download failed [%d]: %s", resp.status_code, url)
                return None
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("downloaded: %s -> %s", url, filepath)
            return str(filepath)
        except requests.RequestException as e:
            logger.warning("download error: %s — %s", url, e)
            return None
        except Exception:
            return None

    def download_batch(self, urls: list[str]) -> list[str]:
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.download_one, u): u for u in urls}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results
