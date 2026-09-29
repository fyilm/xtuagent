"""LLM 工厂：仅智谱 GLM。"""

import logging

from ..core.config import settings
from ..core.exceptions import ConfigError

logger = logging.getLogger(__name__)

# .env / .env.example 中的占位值，出现即视为「未配置」
PLACEHOLDER_KEYS = {
    "",
    "your_zhipu_api_key_here",
    "your_api_key",
    "sk-xxx",
    "changeme",
}


def validate_api_key() -> str:
    """校验 Key 是否可用，返回去除空白后的 Key。失败时抛 ConfigError。"""
    key = (settings.zhipu_api_key or "").strip()
    if key in PLACEHOLDER_KEYS:
        raise ConfigError(
            "ZHIPU_API_KEY 未配置或仍是占位符，请在项目根目录 .env 中填入真实 Key"
            "（申请地址：https://open.bigmodel.cn/ ）"
        )
    if len(key) < 16:
        raise ConfigError("ZHIPU_API_KEY 长度异常，请确认复制完整")
    return key


def create_llm():
    """创建智谱 ChatModel（LangChain 接口）。"""
    api_key = validate_api_key()

    from langchain_community.chat_models import ChatZhipuAI

    logger.info("使用智谱模型：%s", settings.zhipu_model)
    return ChatZhipuAI(
        api_key=api_key,
        model=settings.zhipu_model,
        temperature=settings.llm_temperature,
    )
