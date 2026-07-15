import logging
from typing import Optional

from .config import config, LLMProvider

logger = logging.getLogger(__name__)


def get_chat_model(provider: Optional[LLMProvider] = None):
    provider = provider or config.llm_provider

    if provider == "zhipu":
        try:
            from langchain_community.chat_models import ChatZhipuAI
        except ImportError:
            raise ImportError("请安装 langchain-community")

        api_key = config.zhipu_api_key
        if not api_key:
            raise ValueError("ZHIPU_API_KEY 未设置")
        logger.info("using ZhipuAI model: %s", config.zhipu_model)
        return ChatZhipuAI(
            api_key=api_key,
            model=config.zhipu_model,
            temperature=0.3,
        )

    elif provider == "qianfan":
        import os
        os.environ.setdefault("QIANFAN_ACCESS_KEY", config.qianfan_api_key)
        os.environ.setdefault("QIANFAN_SECRET_KEY", config.qianfan_secret_key)

        try:
            from langchain_community.chat_models import QianfanChatEndpoint
        except ImportError:
            raise ImportError("请安装 langchain-community")

        if not config.qianfan_api_key or not config.qianfan_secret_key:
            raise ValueError("QIANFAN_ACCESS_KEY / QIANFAN_SECRET_KEY 未设置")
        logger.info("using Qianfan model: %s", config.qianfan_model)
        return QianfanChatEndpoint(
            model=config.qianfan_model,
            temperature=0.3,
        )

    else:
        raise ValueError(f"unsupported provider: {provider}")
