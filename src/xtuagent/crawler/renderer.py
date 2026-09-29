"""浏览器渲染兜底：用无头 Chromium 抓取 JS 渲染页 / 绕过简单的 WAF 拦截。

为什么需要它：校内有一批子站用普通 HTTP 请求拿不到内容，两类原因——

1. **JS 单页应用**：首页只有几百字节的骨架，正文由前端异步加载
   （如图书馆 `lib.xtu.edu.cn`，首页仅 2.5KB）；
2. **WAF 拦截**：直接返回 `403 Forbidden`（Apache 默认错误页），
   部分站点对非浏览器客户端有额外校验。

交给真实浏览器渲染后，这两类页面通常都能拿到内容。

默认**不启用**：渲染比直接请求慢一到两个数量级，只在必要时才用。
启用方式见 `scripts/crawl.py --render`。

注意：本模块需要 `playwright` 且本机已安装 Chromium。沙箱/受限网络下
浏览器进程可能被拦截（表现为 `ERR_CONNECTION_CLOSED`），此时应在本机运行。
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 渲染后正文仍然短于该阈值，说明这个页面确实没内容（而不是没渲染出来）
MIN_RENDERED_HTML = 800


def _find_chromium() -> Optional[str]:
    """定位本机 Chromium/Chrome 可执行文件。

    优先用 Playwright 自带的浏览器；找不到再退回系统 Chrome/Edge
    （避免为了跑一次爬虫去下载 150MB 浏览器）。
    """
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        playwright_dir = Path(local) / "ms-playwright"
        if playwright_dir.exists():
            for pattern in ("chromium-*/chrome-win64/chrome.exe",
                            "chromium-*/chrome-win/chrome.exe"):
                candidates.extend(sorted(playwright_dir.glob(pattern), reverse=True))
    for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.append(Path(base) / "Google/Chrome/Application/chrome.exe")
        candidates.append(Path(base) / "Microsoft/Edge/Application/msedge.exe")
    for path in candidates:
        if path and Path(path).exists():
            return str(path)
    return None


class Renderer:
    """无头浏览器封装。惰性启动，用完显式 close()。"""

    def __init__(self, timeout_ms: int = 40000, wait_ms: int = 2500):
        self.timeout_ms = timeout_ms
        self.wait_ms = wait_ms
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_started(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright  # 延迟导入

        self._playwright = sync_playwright().start()
        exe = _find_chromium()
        kwargs = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        }
        if exe:
            kwargs["executable_path"] = exe
        # 让浏览器复用 shell 的代理设置（requests 会自动用，浏览器不会）
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            kwargs["proxy"] = {"server": proxy}
        self._browser = self._playwright.chromium.launch(**kwargs)
        self._context = self._browser.new_context(user_agent=USER_AGENT)
        self._page = self._context.new_page()
        logger.info("浏览器渲染已启动（%s）", exe or "playwright 默认浏览器")

    def fetch(self, url: str) -> Optional[bytes]:
        """渲染页面并返回**渲染后的 HTML 字节**。

        返回渲染后的 DOM 而非原始响应体，这样后续复用同一套
        `_html_to_text` 抽取逻辑即可，无需为渲染路径另写解析。
        """
        try:
            self._ensure_started()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "浏览器渲染不可用（%s）。请确认已安装 playwright 与本机 Chromium；"
                "若报 ERR_CONNECTION_CLOSED，多为受限网络拦截了浏览器进程，"
                "请改在本机运行。",
                exc,
            )
            return None

        try:
            self._page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            self._page.wait_for_timeout(self.wait_ms)
            html = self._page.content()
            if len(html) < MIN_RENDERED_HTML:
                logger.info("渲染后内容仍过短（%d 字节）：%s", len(html), url)
            return html.encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("渲染失败 %s：%s", url, exc)
            return None

    def close(self) -> None:
        for obj, closer in (
            (self._context, lambda o: o.close()),
            (self._browser, lambda o: o.close()),
            (self._playwright, lambda o: o.stop()),
        ):
            if obj is None:
                continue
            try:
                closer(obj)
            except Exception:  # noqa: BLE001, S110
                pass
        self._page = self._context = self._browser = self._playwright = None
