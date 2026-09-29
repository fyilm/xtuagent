"""文档加载测试：同一页面因多个 URL 别名被落盘成多份时的去重。"""

from xtuagent.ingest.loader import _content_key, load_documents


class TestLoaderDedup:
    def test_identical_content_loaded_once(self, tmp_path):
        """同一页面内容以不同文件名落盘，只应加载一次（保留首个文件名）。"""
        (tmp_path / "a_wxy_xtu_edu_cn.txt").write_text(
            "文学与新闻学院 首页 学院概况 学院简介", encoding="utf-8"
        )
        # 仅空白/换行不同，应视为同一文档
        (tmp_path / "b_wxy_xtu_edu_cn_index_htm.txt").write_text(
            "文学与新闻学院\n首页   学院概况\n\n学院简介", encoding="utf-8"
        )
        (tmp_path / "c_other.txt").write_text(
            "湘潭大学图书馆开放时间为每日八点至二十二点。", encoding="utf-8"
        )

        docs = load_documents(tmp_path)

        assert len(docs) == 2
        names = {d.metadata["source_file"] for d in docs}
        assert names == {"a_wxy_xtu_edu_cn.txt", "c_other.txt"}

    def test_different_content_is_kept(self, tmp_path):
        (tmp_path / "one.txt").write_text("转专业管理办法第一条。", encoding="utf-8")
        (tmp_path / "two.txt").write_text("学分绩点计算规则第二条。", encoding="utf-8")

        docs = load_documents(tmp_path)

        assert len(docs) == 2

    def test_content_key_ignores_whitespace(self):
        assert _content_key("a b\nc") == _content_key("a  b c")
        assert _content_key("abc") != _content_key("abd")

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_documents(tmp_path / "not-exist") == []
