"""从本地文件加载 Cookie，用于抓取需要登录的页面。

设计原则：**凭据只留在用户本机**。使用者从自己的浏览器导出 Cookie 存成
文件，由爬虫读取；工具本身不收集、不上传、不记录任何账号密码。

支持两种常见导出格式：

1. Netscape / `cookies.txt`（浏览器扩展「Get cookies.txt」、`curl -c` 的产物）::

       #HttpOnly_.xtu.edu.cn	TRUE	/	TRUE	1735689600	SESSION	abc123

2. JSON 数组（浏览器开发者工具里复制 `document.cookie` 后人工整理，
   或 Playwright/Puppeteer 导出的 `storageState` 片段）::

       [{"name": "SESSION", "value": "abc123", "domain": ".xtu.edu.cn"}]

安全提示：Cookie 等同于登录态，泄露即可被冒充。请勿提交到版本库
（`.gitignore` 已忽略 `*.cookies.txt`），用完可随时删除。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_NETSCAPE_HEADER = "# Netscape HTTP Cookie File"


def load_cookies(path: Optional[str]) -> Dict[str, str]:
    """读取 Cookie 文件，返回 `{name: value}`；失败时返回空字典。"""
    if not path:
        return {}
    filepath = Path(path)
    if not filepath.exists():
        logger.warning("Cookie 文件不存在：%s", filepath)
        return {}

    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}

    if text.lstrip().startswith(("[", "{")):
        return _from_json(text)
    return _from_netscape(text)


def _from_json(text: str) -> Dict[str, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Cookie JSON 解析失败：%s", exc)
        return {}

    # 兼容 Playwright storageState：{"cookies": [...]}
    if isinstance(data, dict) and isinstance(data.get("cookies"), list):
        data = data["cookies"]
    if not isinstance(data, list):
        logger.warning("Cookie JSON 结构无法识别（应为数组或 {cookies: [...]}）")
        return {}

    cookies: Dict[str, str] = {}
    for item in data:
        if isinstance(item, dict) and item.get("name"):
            cookies[str(item["name"])] = str(item.get("value", ""))
    logger.info("已从 JSON 加载 %d 个 Cookie", len(cookies))
    return cookies


def _from_netscape(text: str) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            continue
        line = line.removeprefix("#HttpOnly_")
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        name, value = parts[5].strip(), parts[6].strip()
        if name:
            cookies[name] = value
    if cookies:
        logger.info("已从 Netscape 格式加载 %d 个 Cookie", len(cookies))
    else:
        logger.warning("Cookie 文件格式无法识别（既不是 JSON 也不是 Netscape 格式）")
    return cookies
