"""附件下载：把爬到的附件链接落地为本地文件。

关键点：很多附件链接是 CMS 的**间接下载入口**（如博达的
`system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=…&wbfileid=…`），
URL 里既没有文件名也没有后缀。因此不能只按 URL 路径命名：
- 旧实现只取 `urlparse(url).path` 当文件名，同站所有附件都得到
  `system__content_download.jsp.pdf` 这**同一个**名字，结果 121 个附件
  只能存下 1 个，其余全被当作「已存在」跳过；
- 也无法区分到底是 pdf / doc / xls。

正确做法：下载后优先从 `Content-Disposition` 解析**服务端给出的真实文件名**，
用附件 ID 兜底保证唯一，并校验内容确实是附件（而非登录页/错误页）。
"""

import contextlib
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)

# 附件内容嗅探：文件头魔数 → 后缀
_MAGIC = [
    (b"%PDF-", ".pdf"),
    (b"PK\x03\x04", None),  # docx/xlsx/pptx/zip，需再看内部结构，交给扩展名兜底
    (b"\xd0\xcf\x11\xe0", ".doc"),  # 老版 doc/xls/ppt
    (b"Rar!\x1a\x07", ".rar"),
]

# 从 Content-Disposition 里取文件名（含 filename*=UTF-8''… 形式）
_CD_FILENAME = re.compile(r"filename\*?\s*=\s*(?:UTF-8''|utf-8'')?\"?([^\";\r\n]+)", re.I)


def _safe_filename(name: str) -> str:
    """去掉路径分隔符与非法字符，限制长度。"""
    name = unquote(name).strip().strip('"')
    name = re.split(r"[\\/]", name)[-1]
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    return name[:150] if len(name) > 150 else name


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

    @staticmethod
    def _unique_suffix(url: str) -> str:
        """从 URL 提取能区分不同附件的标识（优先 wbfileid 之类的 ID）。

        间接下载入口的 path 全都一样，必须靠查询参数里的 ID 区分，
        否则所有附件会互相覆盖。
        """
        parsed = urlparse(url)
        query = parsed.query.lower()
        for key in ("wbfileid", "fileid", "attachid", "downloadid"):
            match = re.search(rf"{key}=([^&]+)", query)
            if match:
                return f"{key}{match.group(1)}"
        digest = hashlib.blake2b(url.encode("utf-8"), digest_size=6).hexdigest()
        return digest

    def _derive_filename(self, url: str, content_disposition: str = "") -> str:
        """确定落盘文件名：服务端文件名 > URL 路径名 > URL 指纹。"""
        # 1) 服务端在 Content-Disposition 里给的文件名最可靠
        match = _CD_FILENAME.search(content_disposition or "")
        if match:
            name = _safe_filename(match.group(1))
            if name and "." in name:
                return name

        # 2) URL 路径里带真实后缀的（如 xxx.pdf）
        parsed = urlparse(url)
        path_name = _safe_filename(parsed.path)
        if path_name and re.search(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|md|txt)$", path_name, re.I):
            return path_name

        # 3) 间接入口：用「站点 + 附件ID」拼一个稳定唯一的名字
        host = (parsed.hostname or "unknown").replace(".", "_")
        stem = _safe_filename(Path(parsed.path).stem or "file")
        return f"{host}_{stem}_{self._unique_suffix(url)}"

    @staticmethod
    def _sniff_extension(head: bytes) -> Optional[str]:
        for magic, ext in _MAGIC:
            if head.startswith(magic):
                return ext
        return None

    def download_one(self, url: str) -> Optional[str]:
        try:
            resp = self._session.get(
                url, timeout=self.timeout, allow_redirects=True, stream=True
            )
            if resp.status_code != 200:
                logger.warning("download failed [%d]: %s", resp.status_code, url)
                return None

            # 先读一小段用于嗅探，避免把整站错误页/登录页当附件存下来
            head = next(resp.iter_content(chunk_size=4096), b"")
            sniffed = self._sniff_extension(head)
            content_type = resp.headers.get("Content-Type", "")
            if sniffed is None and "html" in content_type.lower():
                logger.warning("skip non-attachment (html) [%s]: %s", content_type, url)
                return None

            disposition = resp.headers.get("Content-Disposition", "")
            filename = self._derive_filename(url, disposition)
            if sniffed and not filename.lower().endswith(sniffed):
                filename += sniffed
            filepath = self.output_dir / filename

            if filepath.exists() and filepath.stat().st_size > 0:
                logger.info("skip existing: %s", filepath.name)
                return str(filepath)

            with open(filepath, "wb") as f:
                if head:
                    f.write(head)
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size = filepath.stat().st_size
            if size == 0:
                logger.warning("empty file, removed: %s", filepath.name)
                # 沙箱/安全删除钩子可能把 unlink 劫持为回收站操作并使其中止
                with contextlib.suppress(OSError):
                    filepath.unlink()
                return None
            logger.info("downloaded: %s -> %s (%d bytes)", url, filepath.name, size)
            return str(filepath)
        except requests.RequestException as e:
            logger.warning("download error: %s — %s", url, e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("download unexpected error: %s — %s", url, e)
            return None

    def download_batch(self, urls: list[str]) -> list[str]:
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.download_one, u): u for u in urls}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("batch download task failed: %s", exc)
                    continue
                if result:
                    results.append(result)
        return results
