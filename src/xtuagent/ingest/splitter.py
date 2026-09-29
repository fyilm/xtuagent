"""语义分块：RecursiveCharacterTextSplitter 中文优化，**按模型 token 计数**。

为什么必须按 token 而不是按字符计数（重要）
-------------------------------------------
bge-base-zh 的 ``max_position_embeddings`` 是 **512**，而中文大约「1 字 1 token」。
早先的 ``length_function=len`` 数的是**字符**，于是 ``chunk_size=800`` 实际是
800 个 token，**超出上限 56%**，超出的部分被 tokenizer 静默截断。实测：

============================  ==========
token 数（中位 / 最大）        532 / 773
超过 512 token 的块占比        **52.0%**
被截断块平均编码的内容比例    仅前 77.6%
============================  ==========

也就是说，**一半以上的块，尾部两成内容根本没参与向量匹配**——答案恰好落在
块尾时，这个块压根不会被检索到。改用模型真实 tokenizer 计数即可彻底规避。

> 注意：截断只影响「**召回**」，不影响「**生成**」。完整块文本仍存放在索引里，
> 命中后原样交给 LLM，所以症状是**漏召**而不是内容缺失。这也是它长期不易被
> 察觉的原因——答案看起来总是"全对"，只是有些问题干脆问不出结果。

`chunk_size` / `chunk_overlap` 的单位因此从「字符」变为「**token**」。
"""

import logging
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.config import settings

logger = logging.getLogger(__name__)

# 中文优先的分隔符顺序：段落 → 换行 → 中文句读 → 西文句读 → 空格 → 字符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]


@lru_cache(maxsize=200_000)
def _encode_len(text: str) -> int:
    """走模型的 tokenizer 数 token（含缓存）。"""
    # 延迟导入：避免 import 本模块时就拉起 391MB 的模型
    from ..rag.embeddings import EmbeddingService

    tokenizer = EmbeddingService.instance().model.tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))


def model_token_length(text: str) -> int:
    """文本长度 = 模型 tokenizer 下的 token 数（不含 [CLS] / [SEP]）。

    递归切分会对候选片段**反复**调用本函数，所以内容级缓存是必需的；
    没有缓存时整库切分会显著变慢。
    """
    if not text:
        return 0
    return _encode_len(text)


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    length_function=None,
):
    """构造分块器。

    参数单位是 **token**（`chunk_size` / `chunk_overlap`）。默认用嵌入模型的
    真实 tokenizer 计数；单元测试可注入简单计数器以避免加载模型。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=SEPARATORS,
        length_function=length_function or model_token_length,
        is_separator_regex=False,
    )


def split_documents(
    documents: list,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    length_function=None,
) -> list:
    """分块。参数显式传入，不修改全局配置，便于同进程内做多组参数对比实验。"""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = create_splitter(size, overlap, length_function)
    chunks = splitter.split_documents(documents)
    logger.info(
        "分块完成：%d 个文本块（大小 %d / 重叠 %d，单位=token）",
        len(chunks),
        size,
        overlap,
    )
    return chunks
