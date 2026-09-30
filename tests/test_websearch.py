"""联网检索模块单元测试。

全部走桩 HTTP，不打真实网络、不消耗搜索额度。
"""

import httpx
import pytest

from xtuagent.core.config import settings
from xtuagent.core.exceptions import ConfigError, WebSearchError
from xtuagent.rag import websearch
from xtuagent.rag.websearch import CONTENT_LIMIT, WebDoc, build_web_context, web_search


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload) if not isinstance(payload, str) else payload

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setattr(settings, "zhipu_api_key", "test_key_0123456789abcdef")


def _patch_post(monkeypatch, response):
    """把 httpx.post 换成桩，并记录收到的请求，返回记录列表。"""
    calls = []

    def _post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(websearch.httpx, "post", _post)
    return calls


SAMPLE = {
    "search_result": [
        {
            "title": "校园网-湘潭大学",
            "link": "https://www.xtu.edu.cn/xysh/ggfw/xyw.htm",
            "media": "湘潭大学",
            "content": "湘潭大学校园网提供免费 VPN 服务，在公网可共享校园网资源。",
            "publish_date": "",
        },
        {
            "title": "VPN 使用指南",
            "link": "https://example.com/vpn",
            "media": "原创力文档",
            "content": "登录学校邮箱后下载 VPN 客户端，使用学号登录。",
            "publish_date": "2017-07-29",
        },
    ]
}


def test_web_search_parses_structured_results(monkeypatch):
    _patch_post(monkeypatch, _Response(SAMPLE))

    docs = web_search("校园网 VPN")

    assert len(docs) == 2
    assert docs[0].title == "校园网-湘潭大学"
    assert docs[0].link.endswith("xyw.htm")
    assert docs[0].label == "湘潭大学"
    assert docs[1].publish_date == "2017-07-29"


def test_web_search_sends_engine_and_count(monkeypatch):
    calls = _patch_post(monkeypatch, _Response(SAMPLE))

    web_search("校园网 VPN")

    assert calls[0]["url"] == websearch.WEB_SEARCH_URL
    assert calls[0]["json"]["search_engine"] == settings.web_search_engine
    assert calls[0]["json"]["count"] == settings.web_search_count
    assert calls[0]["headers"]["Authorization"].startswith("Bearer ")


def test_web_search_blank_query_short_circuits_without_http(monkeypatch):
    calls = _patch_post(monkeypatch, _Response(SAMPLE))

    assert web_search("   ") == []
    assert calls == [], "空查询不该发出任何请求"


def test_web_search_raises_on_non_200(monkeypatch):
    _patch_post(monkeypatch, _Response("rate limited", status_code=429))

    with pytest.raises(WebSearchError) as err:
        web_search("校园网 VPN")

    assert "429" in str(err.value)


def test_web_search_raises_on_network_error(monkeypatch):
    _patch_post(monkeypatch, httpx.ConnectError("connection refused"))

    with pytest.raises(WebSearchError):
        web_search("校园网 VPN")


def test_web_search_raises_on_invalid_json(monkeypatch):
    _patch_post(monkeypatch, _Response("<html>502</html>"))

    with pytest.raises(WebSearchError):
        web_search("校园网 VPN")


def test_web_search_requires_real_api_key(monkeypatch):
    monkeypatch.setattr(settings, "zhipu_api_key", "your_zhipu_api_key_here")
    calls = _patch_post(monkeypatch, _Response(SAMPLE))

    with pytest.raises(ConfigError):
        web_search("校园网 VPN")
    assert calls == []


def test_web_search_skips_entries_without_content(monkeypatch):
    payload = {
        "search_result": [
            {"title": "空壳", "link": "https://a.com", "media": "A", "content": "   "},
            {"title": "有货", "link": "https://b.com", "media": "B", "content": "正文"},
        ]
    }
    _patch_post(monkeypatch, _Response(payload))

    docs = web_search("x")

    assert [d.title for d in docs] == ["有货"]


def test_web_search_tolerates_missing_fields(monkeypatch):
    _patch_post(monkeypatch, _Response({"search_result": [{"content": "只有正文"}]}))

    docs = web_search("x")

    assert docs[0].content == "只有正文"
    assert docs[0].title == ""
    assert docs[0].label == "网络来源"


def test_web_search_truncates_long_content(monkeypatch):
    long_text = "校" * (CONTENT_LIMIT + 500)
    _patch_post(
        monkeypatch,
        _Response({"search_result": [{"content": long_text, "link": "https://a.com"}]}),
    )

    docs = web_search("x")

    assert len(docs[0].content) == CONTENT_LIMIT


def test_web_search_handles_empty_result_list(monkeypatch):
    _patch_post(monkeypatch, _Response({"search_result": []}))

    assert web_search("校园网 VPN") == []


def test_web_search_truncates_to_configured_count(monkeypatch):
    """对模糊问题该 API 会多意图拆解，返回条数可能超过 count，必须自己截断。"""
    payload = {
        "search_result": [
            {"content": f"正文{i}", "link": f"https://a.com/{i}"} for i in range(10)
        ]
    }
    _patch_post(monkeypatch, _Response(payload))

    docs = web_search("长沙天气", count=3)

    assert len(docs) == 3
    assert [d.content for d in docs] == ["正文0", "正文1", "正文2"]


def test_label_falls_back_to_hostname():
    doc = WebDoc(title="t", link="https://www.xtu.edu.cn/a/b.htm", media="", content="c")
    assert doc.label == "www.xtu.edu.cn"


def test_build_web_context_numbers_and_includes_link():
    docs = [
        WebDoc(title="A", link="https://a.com", media="站点A", content="正文A"),
        WebDoc(
            title="B",
            link="https://b.com",
            media="站点B",
            content="正文B",
            publish_date="2020-01-01",
        ),
    ]

    ctx = build_web_context(docs)

    assert "[1] 标题：A" in ctx
    assert "[2] 标题：B" in ctx
    assert "链接：https://a.com" in ctx
    assert "发布时间：2020-01-01" in ctx
    assert "正文：正文A" in ctx
