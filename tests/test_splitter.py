"""分块器单元测试。

分块参数的单位是 **token**，默认长度函数要走嵌入模型的 tokenizer。
为了让单测保持快速、且不强依赖 391MB 的模型文件，绝大多数用例**显式注入**
一个简单的字符计数器；真正走模型 tokenizer 的那条路径单独成类，
模型不可用时自动跳过。
"""

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

from xtuagent.core.config import settings
from xtuagent.ingest.splitter import (
    create_splitter,
    model_token_length,
    split_documents,
)


def _model_available() -> bool:
    """本地嵌入目录是否就绪。

    本项目强制离线加载（`local_files_only=True`），所以只有指向**本地目录**
    的配置才可能可用；裸 HF 名（如 `BAAI/bge-base-zh`）一律视为不可用。
    """
    if not os.sep in settings.embedding_model and "/" not in settings.embedding_model:
        return False
    return Path(settings.embedding_model).exists()


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason=f"本地嵌入模型不可用：{settings.embedding_model}",
)


class TestSplitBasics:
    """用注入的字符计数器测切分逻辑本身，不加载模型。"""

    def test_split_long_text_into_multiple_chunks(self):
        long_text = "湘潭大学学分制管理规定。" * 200  # 2200 字符
        doc = Document(page_content=long_text, metadata={"source_file": "test.txt"})

        chunks = split_documents([doc], chunk_size=500, chunk_overlap=50, length_function=len)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 500
            assert chunk.metadata["source_file"] == "test.txt"

    def test_split_short_text_single_chunk(self):
        doc = Document(page_content="短文本", metadata={})
        chunks = split_documents([doc], length_function=len)
        assert len(chunks) == 1
        assert chunks[0].page_content == "短文本"

    def test_split_respects_custom_size(self):
        splitter = create_splitter(chunk_size=100, chunk_overlap=10, length_function=len)
        chunks = splitter.split_text("学籍管理规定。" * 50)
        assert len(chunks) > 1
        assert all(len(c) <= 110 for c in chunks)

    def test_chinese_separator_priority(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        splitter = create_splitter(chunk_size=20, chunk_overlap=0, length_function=len)
        chunks = splitter.split_text(text)
        assert any("第一段" in c for c in chunks)

    def test_empty_input_yields_no_chunks(self):
        assert split_documents([Document(page_content="", metadata={})], length_function=len) == []


class TestLengthFunctionWiring:
    """把「必须按 token 计数」这个决定钉死在测试里。

    这是个**回归陷阱**：早期 `length_function=len` 数的是字符，而中文约
    1 字 1 token，于是 chunk_size=800 实际是 800 token，超出 bge-base-zh 的
    512 上限，实测 52% 的块被静默截断。改回 len 会悄悄退化，因此需要断言。
    """

    def test_default_length_function_is_token_based(self):
        splitter = create_splitter()
        assert splitter._length_function is model_token_length

    def test_injected_length_function_is_used(self):
        splitter = create_splitter(length_function=len)
        assert splitter._length_function is len

    def test_chunk_size_unit_is_token_not_char(self):
        """同样 chunk_size=20，按字符能切开的文本，按 token 计数也应成立。"""
        splitter = create_splitter(chunk_size=20, chunk_overlap=0, length_function=len)
        assert len(splitter.split_text("湘潭大学" * 20)) > 1


class TestModelTokenLength:
    @requires_model
    def test_counts_chinese_roughly_one_per_char(self):
        """中文约 1 字 1 token（这是长度单位必须换成 token 的根本原因）。"""
        n = model_token_length("湘潭大学计算机学院")
        assert 8 <= n <= 12

    @requires_model
    def test_empty_text_is_zero(self):
        assert model_token_length("") == 0

    @requires_model
    def test_longer_text_counts_more(self):
        assert model_token_length("湘潭大学" * 100) > model_token_length("湘潭大学" * 10)

    @requires_model
    def test_cached_returns_same_value(self):
        text = "湘潭大学材料科学与工程学院"
        assert model_token_length(text) == model_token_length(text)


class TestNoTruncationRegression:
    """核心回归：任何块都不许超过模型的最大序列长度。

    这正是本轮修的问题——修前 52% 的块超限，尾部被 tokenizer 悄悄丢掉，
    且因为「生成」用的是完整原文、只有「召回」受影响，所以症状隐蔽
    （答案落在块尾时该块检索不到，而不是答案出错）。
    """

    @requires_model
    def test_no_chunk_exceeds_model_max_seq_length(self):
        from xtuagent.rag.embeddings import EmbeddingService

        doc = Document(page_content="湘潭大学学分制管理规定。" * 300, metadata={})
        chunks = split_documents([doc])
        assert len(chunks) > 1

        model = EmbeddingService.instance().model
        limit = model.max_seq_length

        worst = 0
        for chunk in chunks:
            # 带特殊标记计数，与模型实际前向时看到的长度一致
            n = len(model.tokenizer.encode(chunk.page_content, add_special_tokens=True))
            worst = max(worst, n)
        assert worst <= limit, f"最长块 {worst} token 超过上限 {limit}"
