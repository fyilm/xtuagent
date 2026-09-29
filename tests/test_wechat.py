"""微信公众号采集（scripts/crawl_wechat.py）的离线单测。

全部离线：只测解析、URL 还原、压缩、命名这些纯函数，
不发任何网络请求——搜狗/微信的反爬策略会变，测试必须稳定可跑。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import crawl_wechat as cw  # noqa: E402


# 取自真实搜索结果页的结构（已裁剪），关键点是：
#   1. 公众号名在 <div class="s-p"> 的首个 span，不是 <a class="account">
#   2. 发布日期由 JS 写入 document.write(timeConvert('...'))
#   3. 标题里的关键词被 <em><!--red_beg--> 包裹
SEARCH_HTML = """
<ul class="news-list">
<li id="sogou_vr_11002601_box_0" d="ab735a25-6bee54fc">
  <div class="img-box"><a data-z="art" href="/link?url=AAA&amp;type=2"><img src="//x.jpg"></a></div>
  <div class="txt-box">
    <h3><a target="_blank" href="/link?url=AAA&amp;type=2" id="sogou_vr_11002601_title_0"><em><!--red_beg-->湘潭大学<!--red_end--></em><em><!--red_beg-->计算机学院<!--red_end--></em>陈志文副教授,硕士生导师,欢迎报考</a></h3>
    <p class="txt-info" id="sogou_vr_11002601_summary_0"> 副教授 &amp; 硕士生导师 <em>湘潭大学</em> </p>
    <div class="s-p"><span class="all-time-y2">湘研</span><span class="s2"><script>document.write(timeConvert('1772897135'))</script></span></div>
  </div>
</li>
<li id="sogou_vr_11002601_box_1" d="cd12-9f88">
  <div class="txt-box">
    <h3><a target="_blank" href="/link?url=BBB&amp;type=2" id="sogou_vr_11002601_title_1">湘潭大学化学学院有机化学教研室招聘</a></h3>
    <p class="txt-info" id="sogou_vr_11002601_summary_1">招聘启事</p>
    <div class="s-p"><span class="all-time-y2">X-MOL资讯</span><span class="s2">2024-04-03</span></div>
  </div>
</li>
</ul>
"""

# 搜狗 link 页把真实地址拆成多段拼接，这里含一个 @ 占位符，需被移除
LINK_PAGE_HTML = """
<meta content="always" name="referrer">
<script>
    (new Image()).src = 'https://weixin.sogou.com/approve?uuid=xxx&token=yyy&from=inner';
    setTimeout(function () {
        var url = '';
        url += 'https://mp.';
        url += 'weixin.qq.c';
        url += 'om/s?src=11';
        url += '&timestamp=';
        url += '1790677247&';
        url += 'ver=6996&si';
        url += 'gnature=c4X';
        url += 'tL7f5yWA*L9';
        url += 'oldVBI-aT-l';
        url += '@dI88EM0Uyj8';
        url += '&new=1';
        url.replace("@", "");
        window.location.replace(url)
    },100);
</script>
"""


class TestParseSearchResults:
    def test_parses_both_items(self):
        items = cw.parse_search_results(SEARCH_HTML, "湘潭大学 计算机学院")
        assert len(items) == 2

    def test_strips_highlight_comments_from_title(self):
        items = cw.parse_search_results(SEARCH_HTML, "kw")
        assert items[0].title.startswith("湘潭大学计算机学院陈志文副教授")
        assert "red_beg" not in items[0].title
        assert "<em>" not in items[0].title

    def test_account_from_s_p_span(self):
        """公众号名在 <div class="s-p"> 的首个 span，不是 <a class="account">。"""
        items = cw.parse_search_results(SEARCH_HTML, "kw")
        assert items[0].account == "湘研"
        assert items[1].account == "X-MOL资讯"

    def test_date_from_timeconvert(self):
        items = cw.parse_search_results(SEARCH_HTML, "kw")
        # timeConvert('1772897135') → 2026-03-07（本地时区）
        assert items[0].date.startswith("2026-03-")
        # 没有 timeConvert 时原样保留
        assert items[1].date == "2024-04-03"

    def test_sogou_url_is_absolutized(self):
        items = cw.parse_search_results(SEARCH_HTML, "kw")
        assert items[0].sogou_url.startswith("https://weixin.sogou.com/link?url=")
        assert "&amp;" not in items[0].sogou_url

    def test_snippet_html_unescaped(self):
        items = cw.parse_search_results(SEARCH_HTML, "kw")
        assert "&" in items[0].snippet
        assert "&amp;" not in items[0].snippet

    def test_empty_html_returns_empty(self):
        assert cw.parse_search_results("<html></html>", "kw") == []


class TestResolveRealUrl:
    def test_joins_js_fragments(self):
        url = cw.resolve_real_url(LINK_PAGE_HTML)
        assert url is not None
        assert url.startswith("https://mp.weixin.qq.com/s?src=11")

    def test_removes_at_placeholder(self):
        """/link 页的 JS 故意塞 @ 占位符再 replace 掉，还原时必须一并移除。"""
        url = cw.resolve_real_url(LINK_PAGE_HTML)
        assert "@" not in url

    def test_preserves_signature_symbols(self):
        url = cw.resolve_real_url(LINK_PAGE_HTML)
        assert "signature=c4XtL7f5yWA*L9" in url
        assert url.endswith("&new=1")

    def test_returns_none_without_fragments(self):
        assert cw.resolve_real_url("<html>no script here</html>") is None

    def test_returns_none_when_not_http(self):
        assert cw.resolve_real_url("<script>url += 'ftp://x';</script>") is None


class TestCompactHtml:
    def test_removes_script_and_style(self):
        raw = b"<html><script>var a=1;</script><style>.x{}</style><p>hello</p></html>"
        out = cw.compact_html(raw).decode()
        assert "var a=1" not in out
        assert ".x{}" not in out
        assert "<p>hello</p>" in out

    def test_removes_comments(self):
        raw = b"<html><!--red_beg--><p>hi</p></html>"
        assert "red_beg" not in cw.compact_html(raw).decode()

    def test_preserves_article_content(self):
        """正文在 #js_content 的静态 HTML 里，压缩不得伤及。"""
        raw = '<div id="js_content"><p>陈志文 副教授</p></div><script>x</script>'.encode()
        out = cw.compact_html(raw).decode()
        assert 'id="js_content"' in out
        assert "陈志文 副教授" in out

    def test_non_utf8_returned_as_is(self):
        """编码不明的页面原样返回，避免解码/重编码造成乱码。"""
        raw = b"<html>\xd0\xcf\x11\xe0\xa1\xa1</html>"
        assert cw.compact_html(raw) == raw

    def test_shrinks_multiline_script(self):
        raw = b"<html><script>\n" + b"x=1;\n" * 500 + b"</script><p>t</p></html>"
        assert len(cw.compact_html(raw)) < len(raw) / 10


class TestFileNaming:
    def _art(self, title: str, account: str, url: str) -> cw.Article:
        return cw.Article(title=title, account=account, date="", sogou_url=url)

    def test_prefix_and_extension(self):
        a = self._art("标题", "湘研", "https://weixin.sogou.com/link?url=X")
        name = cw._file_name(a)
        assert name.startswith("wechat_湘研_")
        assert name.endswith(".html")

    def test_differs_for_same_title_different_url(self):
        """标题相同但链接不同（不同文章）不能撞名。"""
        n1 = cw._file_name(self._art("同名标题", "A", "https://x/1"))
        n2 = cw._file_name(self._art("同名标题", "A", "https://x/2"))
        assert n1 != n2

    def test_path_separators_removed(self):
        """标题里的斜杠、冒号等不能变成路径分隔符。"""
        a = self._art("a/b:c*d?e", "acc", "https://x/1")
        name = cw._file_name(a)
        assert "/" not in name and "\\" not in name and ":" not in name

    def test_unknown_account_fallback(self):
        a = self._art("标题", "", "https://x/1")
        assert "unknown" in cw._file_name(a)

    def test_long_title_truncated(self):
        a = self._art("标" * 300, "acc", "https://x/1")
        assert len(cw._file_name(a)) < 140


class TestAntiSpider:
    @pytest.mark.parametrize(
        "marker",
        ["请输入验证码", "antispider", "您的访问过于频繁", "环境异常"],
    )
    def test_block_markers_raise(self, marker):
        with pytest.raises(cw.AntiSpiderError):
            cw._check_block(f"<html>{marker}</html>")

    def test_normal_page_passes(self):
        cw._check_block("<html><p>湘潭大学</p></html>")


class _FakeResp:
    def __init__(self, status: int, text: str = "") -> None:
        self.status_code = status
        self.text = text


class _FakeSession:
    """替身 session：固定返回一个响应，不发网络请求。"""

    headers: dict = {}
    cookies: dict = {}

    def __init__(self, resp: _FakeResp) -> None:
        self._resp = resp
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        return self._resp


def _crawler_with(resp: _FakeResp) -> tuple[cw.WeChatCrawler, _FakeSession]:
    c = cw.WeChatCrawler()
    fake = _FakeSession(resp)
    c.session = fake
    return c, fake


class TestRateLimitStatusCodes:
    """搜狗限流时会直接回 403/429，而不是返回 200 再塞验证码。

    早期版本只认后者，导致限流被当成「该关键词无结果」，主循环一路把
    剩余关键词全部烧掉（实测一次静默丢掉 9 个关键词）。
    """

    @pytest.mark.parametrize("code", [403, 429, 503])
    def test_raises_on_rate_limit_code(self, code):
        crawler, _ = _crawler_with(_FakeResp(code))
        with pytest.raises(cw.AntiSpiderError, match=str(code)):
            crawler._get("https://weixin.sogou.com/weixin?query=x")

    def test_200_with_captcha_marker_still_raises(self):
        crawler, _ = _crawler_with(_FakeResp(200, "<html>请输入验证码</html>"))
        with pytest.raises(cw.AntiSpiderError):
            crawler._get("https://weixin.sogou.com/weixin?query=x")

    def test_normal_200_passes(self):
        crawler, fake = _crawler_with(_FakeResp(200, "<html>正常页面</html>"))
        resp = crawler._get("https://weixin.sogou.com/weixin?query=x")
        assert resp.status_code == 200
        assert fake.calls == 1

    def test_non_rate_limit_error_code_not_raised(self):
        """404 之类的常规错误不该被当成反爬。"""
        crawler, _ = _crawler_with(_FakeResp(404, "not found"))
        assert crawler._get("https://x/").status_code == 404


class TestSearchRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        crawler = cw.WeChatCrawler()
        calls = {"n": 0}

        def fake_search(keyword, page=1):
            calls["n"] += 1
            if calls["n"] < 3:
                raise cw.AntiSpiderError("HTTP 403（限流）")
            return [cw.Article(title="t", account="a", date="", sogou_url="u")]

        monkeypatch.setattr(crawler, "search", fake_search)
        out = crawler.search_with_retry("kw", 1, retries=3, cooldown=0)
        assert len(out) == 1
        assert calls["n"] == 3

    def test_raises_after_exhausting_retries(self, monkeypatch):
        crawler = cw.WeChatCrawler()

        def always_blocked(keyword, page=1):
            raise cw.AntiSpiderError("HTTP 403（限流）")

        monkeypatch.setattr(crawler, "search", always_blocked)
        with pytest.raises(cw.AntiSpiderError, match="连续"):
            crawler.search_with_retry("kw", 1, retries=2, cooldown=0)

    def test_request_exception_also_retried(self, monkeypatch):
        import requests as _rq

        crawler = cw.WeChatCrawler()
        calls = {"n": 0}

        def flaky(keyword, page=1):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _rq.ConnectionError("boom")
            return []

        monkeypatch.setattr(crawler, "search", flaky)
        assert crawler.search_with_retry("kw", 1, retries=2, cooldown=0) == []
        assert calls["n"] == 2

    def test_no_retry_when_first_call_ok(self, monkeypatch):
        crawler = cw.WeChatCrawler()
        calls = {"n": 0}

        def ok(keyword, page=1):
            calls["n"] += 1
            return []

        monkeypatch.setattr(crawler, "search", ok)
        crawler.search_with_retry("kw", 1, retries=5, cooldown=0)
        assert calls["n"] == 1


class TestFmtDate:
    def test_valid_timestamp(self):
        assert cw._fmt_date("1772897135").startswith("2026-")

    def test_invalid_returns_input(self):
        assert cw._fmt_date("not-a-number") == "not-a-number"

    def test_out_of_range_returns_input(self):
        assert cw._fmt_date("99999999999999") == "99999999999999"
