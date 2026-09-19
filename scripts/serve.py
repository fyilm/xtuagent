#!/usr/bin/env python
"""启动 Web 服务（FastAPI + 静态前端）。"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xtuagent.core.config import settings
from xtuagent.core.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 XtuAgent Web 服务")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    setup_logging()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  XtuAgent 服务启动中…… 访问地址: {url}\n")

    if not args.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(
        "xtuagent.serving.app:app",
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
