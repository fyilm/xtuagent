"""日志配置：控制台 + 文件轮转。"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import settings

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_configured = False


def setup_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True

    settings.ensure_dirs()
    root = logging.getLogger()
    root.setLevel(level or settings.log_level)
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "xtuagent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("urllib3", "httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
