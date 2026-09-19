"""领域异常体系。"""


class XtuAgentError(Exception):
    """所有业务异常的基类。"""


class ConfigError(XtuAgentError):
    """配置缺失或非法。"""


class IndexNotReadyError(XtuAgentError):
    """索引文件不存在或未构建。"""


class IndexMetaMismatchError(XtuAgentError):
    """索引元数据与当前配置不匹配（如嵌入模型不一致）。"""


class LLMInvocationError(XtuAgentError):
    """大模型调用失败（重试后仍失败）。"""
