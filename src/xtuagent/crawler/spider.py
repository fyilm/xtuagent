import re
import time
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .site_config import CRAWL_RULES

logger = logging.getLogger(__name__)


def _decode_content(resp: requests.Response) -> str:
    raw = resp.content
    enc = resp.encoding
    match = re.search(rb'charset[\s]*=[\s]*"?([^";\s]+)', raw[:5000])
    if match:
        enc = match.group(1).decode("ascii", errors="ignore").lower()
    if enc in ("gb2312", "gb231280"):
        enc = "gbk"
    candidates = ["gbk", "utf-8"] if enc == "gbk" else ["utf-8", "gbk"]

    best_text, best_score = None, -1
    for e in candidates:
        text = raw.decode(e, errors="replace")
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii == 0:
            score = 0
        else:
            cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            noise = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
            noise += sum(1 for c in text if "\u0370" <= c <= "\u03FF")
            noise += text.count("\ufffd")
            score = cjk / max(non_ascii, 1)
            if noise > 0:
                score /= 1 + noise * 10
        if score > best_score:
            best_score, best_text = score, text
    return (best_text or raw.decode("utf-8", errors="replace")).replace("\ufffd", " ")


class Spider:
    def __init__(
        self,
        max_depth: int = 3,
        max_pages_per_site: int = 80,
        delay: float = 1.0,
        timeout: int = 30,
        allowed_domains: Optional[list[str]] = None,
        file_types: Optional[list[str]] = None,
    ):
        self.max_depth = max_depth
        self.max_pages_per_site = max_pages_per_site
        self.delay = delay
        self.timeout = timeout
        self.allowed_domains = allowed_domains or CRAWL_RULES["allowed_domains"]
        self.file_types = file_types or CRAWL_RULES["file_types"]
        self._visited: set[str] = set()
        self._content_saved: set[str] = set()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def _is_allowed_domain(self, url: str) -> bool:
        hostname = urlparse(url).hostname
        if not hostname:
            return False
        return any(hostname == d or hostname.endswith("." + d) for d in self.allowed_domains)

    def _is_file_link(self, url: str) -> bool:
        lower = url.lower()
        return any(lower.endswith(ft) for ft in self.file_types)

    def _is_content_page(self, url: str) -> bool:
        return bool(re.search(r"(/info/\d+/\d+\.htm|/content/|/article/|\d+\.htm$|\d+\.html$)", url))

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links = []
        for a in soup.find_all("a", href=True):
            href = str(a["href"]).strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full_url = urljoin(base_url, href)
            full_url = full_url.split("#")[0]
            if not full_url.startswith("http"):
                continue
            if not self._is_allowed_domain(full_url):
                continue
            links.append(full_url)
        return links

    def _text_from_bytes(self, raw: bytes) -> str:
        b = raw
        b = re.sub(rb"<script[^>]*>.*?</script>", b" ", b, flags=re.DOTALL)
        b = re.sub(rb"<style[^>]*>.*?</style>", b" ", b, flags=re.DOTALL)
        b = re.sub(rb"<[^>]+>", b" ", b)
        b = re.sub(rb"&[a-zA-Z]+;", b" ", b)
        b = re.sub(rb"\s+", b" ", b).strip()
        if len(b) < 50:
            return ""

        best_text, best_score = None, -1
        for e in ["gbk", "utf-8", "gb18030"]:
            text = b.decode(e, errors="replace").replace("\ufffd", " ")
            cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            noise = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
            noise += sum(1 for c in text if "\u0370" <= c <= "\u03FF")
            score = cjk / max(1, cjk + noise) if cjk + noise > 0 else 0
            if score > best_score:
                best_score, best_text = score, text
        return best_text or ""

    def crawl(
        self, start_urls: list[str], output_dir: Optional[str] = None
    ) -> tuple[list[str], list[str]]:
        file_links: list[str] = []
        page_links: list[str] = []
        site_counts: dict[str, int] = {}

        for start_url in start_urls:
            queue: deque[tuple[str, int]] = deque()
            queue.append((start_url.rstrip("/"), 0))
            site_key = urlparse(start_url).hostname or start_url
            site_counts[site_key] = 0

            while queue:
                url, depth = queue.popleft()

                if url in self._visited:
                    continue
                if depth > self.max_depth:
                    continue
                if not self._is_allowed_domain(url):
                    continue

                host = urlparse(url).hostname or ""
                count = site_counts.get(host, 0)
                if count >= self.max_pages_per_site:
                    continue

                self._visited.add(url)
                site_counts[host] = count + 1

                if self._is_file_link(url):
                    file_links.append(url)
                    logger.info("found file: %s", url)
                    continue

                try:
                    resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
                    if resp.status_code != 200:
                        continue
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        continue

                    soup = BeautifulSoup(resp.content.decode("utf-8", errors="replace"), "lxml")
                    page_links.append(url)
                    is_content = self._is_content_page(url)
                    marker = "*" if is_content else " "
                    logger.info("%s page: %s (depth=%d)", marker, url, depth)

                    if output_dir and url not in self._content_saved:
                        text = self._text_from_bytes(resp.content)
                        if text and len(text) > 200:
                            self._save_text(text, url, output_dir)
                            self._content_saved.add(url)
                            logger.info("saved: %s (%d chars)", url, len(text))

                    if depth < self.max_depth:
                        child_links = self._extract_links(soup, url)
                        for link in child_links:
                            child_host = urlparse(link).hostname or ""
                            child_count = site_counts.get(child_host, 0)
                            if link not in self._visited and child_count < self.max_pages_per_site:
                                queue.append((link, depth + 1))

                except requests.RequestException as e:
                    logger.warning("request failed: %s", e)
                    continue
                except Exception as e:
                    logger.warning("unexpected error at %s: %s", url, e)
                    continue

                time.sleep(self.delay)

        return page_links, list(set(file_links))

    def _save_text(self, text: str, url: str, output_dir: str) -> None:
        from pathlib import Path
        from urllib.parse import urlparse

        import re as _re

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(url)
        safe_name = _re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\-_]", "_", parsed.netloc + parsed.path)
        safe_name = safe_name[:120] if len(safe_name) > 120 else safe_name
        filepath = path / f"{safe_name}.txt"
        if not filepath.exists():
            filepath.write_text(text, encoding="utf-8")

    @property
    def visited_count(self) -> int:
        return len(self._visited)
