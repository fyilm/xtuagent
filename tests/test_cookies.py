"""Cookie 加载测试：支持 Netscape 与 JSON 两种常见导出格式。"""

import json

from xtuagent.crawler.cookies import load_cookies


class TestNetscapeFormat:
    def test_parses_standard_cookies_txt(self, tmp_path):
        f = tmp_path / "cookies.txt"
        f.write_text(
            "# Netscape HTTP Cookie File\n"
            "#HttpOnly_.xtu.edu.cn\tTRUE\t/\tTRUE\t1735689600\tSESSION\tabc123\n"
            ".xtu.edu.cn\tTRUE\t/\tFALSE\t1735689600\tSERVERID\tnode1\n",
            encoding="utf-8",
        )

        cookies = load_cookies(str(f))

        assert cookies == {"SESSION": "abc123", "SERVERID": "node1"}

    def test_ignores_comment_lines(self, tmp_path):
        f = tmp_path / "cookies.txt"
        f.write_text(
            "# Netscape HTTP Cookie File\n# 这是注释\n",
            encoding="utf-8",
        )

        assert load_cookies(str(f)) == {}


class TestJsonFormat:
    def test_parses_playwright_storage_state(self, tmp_path):
        f = tmp_path / "state.json"
        f.write_text(
            json.dumps({"cookies": [
                {"name": "SESSION", "value": "abc123", "domain": ".xtu.edu.cn"},
                {"name": "SERVERID", "value": "node1", "domain": ".xtu.edu.cn"},
            ]}),
            encoding="utf-8",
        )

        assert load_cookies(str(f)) == {"SESSION": "abc123", "SERVERID": "node1"}

    def test_parses_plain_list(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text(json.dumps([{"name": "A", "value": "1"}]), encoding="utf-8")

        assert load_cookies(str(f)) == {"A": "1"}


class TestRobustness:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_cookies(str(tmp_path / "nope.txt")) == {}

    def test_none_returns_empty(self):
        assert load_cookies(None) == {}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        assert load_cookies(str(f)) == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[{not valid json", encoding="utf-8")

        assert load_cookies(str(f)) == {}


class TestSpiderAppliesCookies:
    def test_cookies_land_in_session(self, tmp_path):
        from xtuagent.crawler.spider import Spider

        spider = Spider(prefer_ipv4=False, cookies={"SESSION": "abc123"})

        assert spider._session.cookies.get("SESSION") == "abc123"


class TestLocalHtmlIngestion:
    """抓不到的站点：用户在浏览器「另存为」后灌入，应能抽出正文。"""

    def test_saved_html_page_is_extracted(self, tmp_path):
        from xtuagent.crawler.text_extractor import TextExtractor
        from scripts.ingest_urls import _text_from_file

        saved = tmp_path / "page.html"
        saved.write_text(
            "<html><head><title>商学院通知-湘潭大学商学院</title></head><body>"
            "<div class='nav'><a>首页</a><a>学院概况</a></div>"
            "<div id='vsb_content'><p>第一条 商学院转专业实施细则，学分绩点要求不低于2.0。</p></div>"
            "</body></html>",
            encoding="utf-8",
        )

        text = _text_from_file(saved, TextExtractor(output_dir=str(tmp_path)))

        assert "学分绩点要求不低于2.0" in text
        assert "学院概况" not in text
