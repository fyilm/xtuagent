"""附件下载命名测试：间接下载入口必须能区分出不同附件。"""

from urllib.parse import parse_qs, urlparse

from xtuagent.crawler.downloader import Downloader


VSB_A = (
    "https://jwc.xtu.edu.cn/system/_content/download.jsp"
    "?urltype=news.DownloadAttachUrl&owner=1732349760&wbfileid=18180674"
)
VSB_B = (
    "https://jwc.xtu.edu.cn/system/_content/download.jsp"
    "?urltype=news.DownloadAttachUrl&owner=1732349760&wbfileid=18180679"
)


class TestFilenameDerivation:
    def test_indirect_endpoints_get_distinct_names(self, tmp_path):
        """回归：旧实现只按 URL path 命名，同站附件全部同名、互相覆盖，121 个只存下 1 个。"""
        dl = Downloader(output_dir=str(tmp_path))

        name_a = dl._derive_filename(VSB_A)
        name_b = dl._derive_filename(VSB_B)

        assert name_a != name_b
        assert "18180674" in name_a
        assert "18180679" in name_b

    def test_content_disposition_wins(self, tmp_path):
        dl = Downloader(output_dir=str(tmp_path))
        disposition = "attachment; filename=\"培养方案2026.pdf\""

        assert dl._derive_filename(VSB_A, disposition) == "培养方案2026.pdf"

    def test_rfc5987_filename_star(self, tmp_path):
        dl = Downloader(output_dir=str(tmp_path))
        disposition = "attachment; filename*=UTF-8''%E5%AD%A6%E7%94%9F%E6%89%8B%E5%86%8C.pdf"

        assert dl._derive_filename(VSB_A, disposition) == "学生手册.pdf"

    def test_direct_url_keeps_suffix(self, tmp_path):
        dl = Downloader(output_dir=str(tmp_path))

        assert dl._derive_filename("https://jwc.xtu.edu.cn/a/plan.pdf") == "plan.pdf"

    def test_unsafe_path_characters_sanitised(self, tmp_path):
        dl = Downloader(output_dir=str(tmp_path))
        name = dl._derive_filename(VSB_A, 'attachment; filename="../../etc/passwd.pdf"')

        assert "/" not in name
        assert ".." not in name

    def test_magic_bytes_sniffing(self):
        assert Downloader._sniff_extension(b"%PDF-1.7\n...") == ".pdf"
        assert Downloader._sniff_extension(b"\xd0\xcf\x11\xe0\xa1\xb1") == ".doc"
        assert Downloader._sniff_extension(b"<html><body>login") is None
