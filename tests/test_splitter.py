"""分块器单元测试。"""

from langchain_core.documents import Document

from xtuagent.ingest.splitter import create_splitter, split_documents


def test_split_long_text_into_multiple_chunks():
    long_text = "湘潭大学学分制管理规定。" * 200
    doc = Document(page_content=long_text, metadata={"source_file": "test.txt"})

    chunks = split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 900
        assert chunk.metadata["source_file"] == "test.txt"


def test_split_short_text_single_chunk():
    doc = Document(page_content="短文本", metadata={})
    chunks = split_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "短文本"


def test_split_respects_custom_size():
    splitter = create_splitter(chunk_size=100, chunk_overlap=10)
    text = "学籍管理规定。" * 50
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 110 for c in chunks)


def test_chinese_separator_priority():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    splitter = create_splitter(chunk_size=20, chunk_overlap=0)
    chunks = splitter.split_text(text)
    assert any("第一段" in c for c in chunks)
