"""LLM 工厂：仅智谱 GLM。"""

import logging

from ..core.config import settings
from ..core.exceptions import ConfigError

logger = logging.getLogger(__name__)


def create_llm():
    """创建智谱 ChatModel（LangChain 接口）。"""
    if not settings.zhipu_api_key:
        raise ConfigError("ZHIPU_API_KEY 未配置，请在 .env 中设置")

    from langchain_community.chat_models import ChatZhipuAI

    logger.info("使用智谱模型：%s", settings.zhipu_model)
    return ChatZhipuAI(
        api_key=settings.zhipu_api_key,
        model=settings.zhipu_model,
        temperature=settings.llm_temperature,
    )
