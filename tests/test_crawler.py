"""爬虫单元测试：断点状态、过滤/优先级规则。

不联网（prefer_ipv4=False 以避免改动全局 socket 解析）。
"""

import json

import pytest

from xtuagent.crawler.spider import Spider


def make_spider(tmp_path=None, resume=True):
    return Spider(
        state_file=str(tmp_path / "state.json") if tmp_path else None,
        resume=resume,
        prefer_ipv4=False,
    )


class TestCrawlState:
    def test_state_roundtrip_keeps_visited_and_pending(self, tmp_path):
        first = make_spider(tmp_path, resume=False)
        first._visited = {"https://jwc.xtu.edu.cn/info/1003/4326.htm"}
        first._pending = [("https://jwc.xtu.edu.cn/x.htm", 1)]
        first.save_state()

        second = make_spider(tmp_path, resume=True)

        assert "https://jwc.xtu.edu.cn/info/1003/4326.htm" in second._visited
        assert ("https://jwc.xtu.edu.cn/x.htm", 1) in second._pending

    def test_pending_queue_is_what_makes_resume_possible(self, tmp_path):
        """只存 visited 会导致续爬时起点被跳过、队列瞬间见底 —— 这是必须回归的点。"""
        first = make_spider(tmp_path, resume=False)
        first._visited = {"https://www.xtu.edu.cn"}  # 起点已访问
        first._pending = [("https://jwc.xtu.edu.cn", 0)]
        first.save_state()

        resumed = make_spider(tmp_path, resume=True)

        assert resumed._pending, "续爬后待爬队列不应为空"

    def test_fresh_run_ignores_existing_state(self, tmp_path):
        first = make_spider(tmp_path, resume=False)
        first._visited = {"https://a.xtu.edu.cn/1"}
        first.save_state()

        fresh = make_spider(tmp_path, resume=False)

        assert fresh._visited == set()
        assert fresh._pending == []

    def test_corrupt_state_is_ignored(self, tmp_path):
        (tmp_path / "state.json").write_text("{ this is not json", encoding="utf-8")

        spider = make_spider(tmp_path, resume=True)

        assert spider._visited == set()
        assert spider._pending == []

    def test_malformed_pending_entries_are_skipped(self, tmp_path):
        (tmp_path / "state.json").write_text(
            json.dumps({"visited": ["https://a.xtu.edu.cn"], "pending": [["u", 1], "bad", [1]]}),
            encoding="utf-8",
        )

        spider = make_spider(tmp_path, resume=True)

        assert spider._pending == [("u", 1)]


class TestUrlRules:
    def test_exclude_patterns_block_media(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_excluded("https://www.xtu.edu.cn/images/a.jpg")
        assert spider._is_excluded("https://www.xtu.edu.cn/video/b.mp4")
        assert not spider._is_excluded("https://jwc.xtu.edu.cn/info/1003/4326.htm")

    def test_priority_patterns_match_content_columns(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_priority("https://jwc.xtu.edu.cn/info/1003/4326.htm")
        assert spider._is_priority("https://xgb.xtu.edu.cn/tzgg/index.htm")
        assert not spider._is_priority("https://www.xtu.edu.cn/")

    def test_content_page_detection(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_content_page("https://news.xtu.edu.cn/info/1032/31470.htm")
        assert not spider._is_content_page("https://www.xtu.edu.cn/")

    def test_domain_allowlist_covers_subdomains_only(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_allowed_domain("https://jwc.xtu.edu.cn/x.htm")
        assert spider._is_allowed_domain("https://xtu.edu.cn/x.htm")
        assert not spider._is_allowed_domain("https://evil-xtu.edu.cn/x.htm")
        assert not spider._is_allowed_domain("https://example.com/x.htm")

    def test_doc_and_docx_are_not_downloaded(self, tmp_path):
        """TextExtractor 无法解析 doc/docx，不应再浪费下载。"""
        spider = make_spider(tmp_path)

        assert not spider._is_file_link("https://jwc.xtu.edu.cn/a.doc")
        assert not spider._is_file_link("https://jwc.xtu.edu.cn/a.docx")
        assert spider._is_file_link("https://jwc.xtu.edu.cn/a.pdf")

    def test_enqueueable_respects_visited_and_site_cap(self, tmp_path):
        spider = make_spider(tmp_path)
        spider._visited = {"https://jwc.xtu.edu.cn/seen.htm"}

        assert not spider._enqueueable("https://jwc.xtu.edu.cn/seen.htm", {})
        assert spider._enqueueable("https://jwc.xtu.edu.cn/new.htm", {"jwc.xtu.edu.cn": 0})

        over_cap = {"jwc.xtu.edu.cn": spider.max_pages_per_site}
        assert not spider._enqueueable("https://jwc.xtu.edu.cn/new.htm", over_cap)


class TestHtmlExtraction:
    """正文抽取：必须剥离导航/页脚，且不能把正文连坐删掉。"""

    def test_nav_and_footer_are_stripped(self, tmp_path):
        spider = make_spider(tmp_path)
        html = """
        <html><head><title>关于转专业的通知-教务处</title></head><body>
          <div class="header"><ul class="nav">
            <li>首页</li><li>通知公告</li><li>规章制度</li><li>下载专区</li>
          </ul></div>
          <div id="vsb_content"><p>第一条 申请转专业的学生须为全日制本科一年级学生。</p></div>
          <div class="footer">版权所有 湘潭大学 湘ICP备12345号</div>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "第一条 申请转专业的学生须为全日制本科一年级学生。" in text
        assert "下载专区" not in text
        assert "湘ICP备" not in text
        assert "关于转专业的通知" in text  # 标题被保留，利于检索

    def test_content_survives_when_nested_in_form(self, tmp_path):
        """回归：博达 CMS 的 #vsb_content 嵌在 <form> 内，删 form 会把正文一起删掉。"""
        spider = make_spider(tmp_path)
        html = """
        <html><body>
          <div class="header"><span>首页</span><span>学校概况</span></div>
          <form>
            <div class="textShow"><div class="textContent">
              <div id="vsb_content"><p>第一条 为规范本科学生转专业工作，制定本办法。</p></div>
            </div></div>
          </form>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "第一条 为规范本科学生转专业工作，制定本办法。" in text
        assert "学校概况" not in text

    def test_falls_back_to_body_with_junk_removed(self, tmp_path):
        """没有正文容器时，回退到 body 并剔除命中导航词元的容器。"""
        spider = make_spider(tmp_path)
        html = """
        <html><body>
          <div class="nav-menu"><a>首页</a><a>联系我们</a></div>
          <div class="page-body"><p>湘潭大学图书馆开放时间为每日八点至二十二点。</p></div>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "湘潭大学图书馆开放时间为每日八点至二十二点。" in text
        assert "联系我们" not in text

    def test_drop_tags_never_include_form(self):
        """form 不能出现在必删标签里，否则正文容器会被连坐删除。"""
        from xtuagent.crawler.spider import _DROP_ALWAYS, _DROP_TAGS

        assert "form" not in _DROP_TAGS
        assert "form" not in _DROP_ALWAYS


class TestHtmlExtraction:
    """正文抽取：必须剥离导航/页脚，且不能把正文连坐删掉。"""

    def test_nav_and_footer_are_stripped(self, tmp_path):
        spider = make_spider(tmp_path)
        html = """
        <html><head><title>关于转专业的通知-教务处</title></head><body>
          <div class="header"><ul class="nav">
            <li>首页</li><li>通知公告</li><li>规章制度</li><li>下载专区</li>
          </ul></div>
          <div id="vsb_content"><p>第一条 申请转专业的学生须为全日制本科一年级学生。</p></div>
          <div class="footer">版权所有 湘潭大学 湘ICP备12345号</div>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "第一条 申请转专业的学生须为全日制本科一年级学生。" in text
        assert "下载专区" not in text
        assert "湘ICP备" not in text
        assert "关于转专业的通知" in text  # 标题被保留，利于检索

    def test_content_survives_when_nested_in_form(self, tmp_path):
        """回归：博达 CMS 的 #vsb_content 嵌在 <form> 内，删 form 会把正文一起删掉。"""
        spider = make_spider(tmp_path)
        html = """
        <html><body>
          <div class="header"><span>首页</span><span>学校概况</span></div>
          <form>
            <div class="textShow"><div class="textContent">
              <div id="vsb_content"><p>第一条 为规范本科学生转专业工作，制定本办法。</p></div>
            </div></div>
          </form>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "第一条 为规范本科学生转专业工作，制定本办法。" in text
        assert "学校概况" not in text

    def test_falls_back_to_body_with_junk_removed(self, tmp_path):
        """没有正文容器时，回退到 body 并剔除命中导航词元的容器。"""
        spider = make_spider(tmp_path)
        html = """
        <html><body>
          <div class="nav-menu"><a>首页</a><a>联系我们</a></div>
          <div class="page-body"><p>湘潭大学图书馆开放时间为每日八点至二十二点。</p></div>
        </body></html>
        """
        text = spider._html_to_text(html)

        assert "湘潭大学图书馆开放时间为每日八点至二十二点。" in text
        assert "联系我们" not in text

    def test_drop_tags_never_include_form(self):
        """form 不能出现在必删标签里，否则正文容器会被连坐删除。"""
        from xtuagent.crawler.spider import _DROP_ALWAYS, _DROP_TAGS

        assert "form" not in _DROP_TAGS
        assert "form" not in _DROP_ALWAYS


class TestAttachmentDetection:
    """附件识别：必须认出博达 CMS 的间接下载入口（URL 里没有 .pdf）。"""

    def test_indirect_download_endpoint_is_a_file_link(self, tmp_path):
        spider = make_spider(tmp_path)
        vsb = (
            "https://jwc.xtu.edu.cn/system/_content/download.jsp"
            "?urltype=news.DownloadAttachUrl&owner=1732349760&wbfileid=18180674"
        )

        assert spider._is_file_link(vsb)

    def test_direct_suffix_still_detected(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_file_link("https://jwc.xtu.edu.cn/a/b/c.pdf")
        assert spider._is_file_link("https://jwc.xtu.edu.cn/x/plan.PDF")

    def test_html_page_is_not_a_file_link(self, tmp_path):
        spider = make_spider(tmp_path)

        assert not spider._is_file_link("https://jwc.xtu.edu.cn/info/1016/4902.htm")

    def test_attachment_is_prioritised(self, tmp_path):
        """附件要优先入队，否则会被普通页面挤掉预算。"""
        spider = make_spider(tmp_path)
        vsb = "https://jwc.xtu.edu.cn/system/_content/download.jsp?wbfileid=1"

        assert spider._is_priority(vsb)


class TestAttachmentDetection:
    """附件识别：必须认出博达 CMS 的间接下载入口（URL 里没有 .pdf）。"""

    def test_indirect_download_endpoint_is_a_file_link(self, tmp_path):
        spider = make_spider(tmp_path)
        vsb = (
            "https://jwc.xtu.edu.cn/system/_content/download.jsp"
            "?urltype=news.DownloadAttachUrl&owner=1732349760&wbfileid=18180674"
        )

        assert spider._is_file_link(vsb)

    def test_direct_suffix_still_detected(self, tmp_path):
        spider = make_spider(tmp_path)

        assert spider._is_file_link("https://jwc.xtu.edu.cn/a/b/c.pdf")
        assert spider._is_file_link("https://jwc.xtu.edu.cn/x/plan.PDF")

    def test_html_page_is_not_a_file_link(self, tmp_path):
        spider = make_spider(tmp_path)

        assert not spider._is_file_link("https://jwc.xtu.edu.cn/info/1016/4902.htm")

    def test_attachment_is_prioritised(self, tmp_path):
        """附件要优先入队，否则会被普通页面挤掉预算。"""
        spider = make_spider(tmp_path)
        vsb = "https://jwc.xtu.edu.cn/system/_content/download.jsp?wbfileid=1"

        assert spider._is_priority(vsb)


class TestQueueDedup:
    def test_same_url_is_enqueued_only_once(self, tmp_path):
        """回归：同一 URL 被多页链接时不应重复入队（深度3会膨胀到两万条）。"""
        spider = make_spider(tmp_path)
        url = "https://jwc.xtu.edu.cn/info/1016/4902.htm"

        assert spider._enqueueable(url, {})
        spider._queued.add(url)
        assert not spider._enqueueable(url, {})

    def test_resume_repopulates_queued_set(self, tmp_path):
        first = make_spider(tmp_path, resume=False)
        first._pending = [("https://jwc.xtu.edu.cn/pending.htm", 1)]
        first.save_state()

        second = make_spider(tmp_path, resume=True)

        assert "https://jwc.xtu.edu.cn/pending.htm" in second._queued
        assert not second._enqueueable("https://jwc.xtu.edu.cn/pending.htm", {})


class TestQueueDedup:
    def test_same_url_is_enqueued_only_once(self, tmp_path):
        """回归：同一 URL 被多页链接时不应重复入队（深度3会膨胀到两万条）。"""
        spider = make_spider(tmp_path)
        url = "https://jwc.xtu.edu.cn/info/1016/4902.htm"

        assert spider._enqueueable(url, {})
        spider._queued.add(url)
        assert not spider._enqueueable(url, {})

    def test_resume_repopulates_queued_set(self, tmp_path):
        first = make_spider(tmp_path, resume=False)
        first._pending = [("https://jwc.xtu.edu.cn/pending.htm", 1)]
        first.save_state()

        second = make_spider(tmp_path, resume=True)

        assert "https://jwc.xtu.edu.cn/pending.htm" in second._queued
        assert not second._enqueueable("https://jwc.xtu.edu.cn/pending.htm", {})
