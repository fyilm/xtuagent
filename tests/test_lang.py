"""提问语言自动识别，以及 `AskRequest` 对 language="auto" 的解析。

背景：产品要求「留学生用英语或日语提问，就用对应语言作答」。
判定放在服务端（`serving/schemas.py`）而非前端——语言决定提示词怎么写，
属产品行为；放后端则 `/api/ask`、`/api/ask/stream`、`/api/agent` 一次生效。
"""

import pytest

from xtuagent.core.lang import detect_language
from xtuagent.serving.schemas import AskRequest


@pytest.mark.parametrize(
    "question, expected",
    [
        ("如何办理休学手续？", "中文"),
        ("绩点是怎么计算的", "中文"),
        ("Where can I download MATLAB?", "英语"),
        ("How do I apply for a scholarship?", "英语"),
        ("図書館の開館時間を教えてください", "日语"),
        ("休学の手続きはどうすればいいですか", "日语"),
        ("도서관 운영 시간이 궁금합니다", "韩语"),
        ("Как оформить академический отпуск?", "俄语"),
    ],
)
def test_detect_language(question, expected):
    assert detect_language(question) == expected


def test_kana_checked_before_han():
    """日文必夹假名，而汉字区间**同时命中中日文**。

    判定顺序必须先假名后汉字，否则「図書館は何時まで」会被误判成中文。
    """
    assert detect_language("図書館は何時まで？") == "日语"


def test_chinese_with_latin_still_chinese():
    """中文里夹英文词（软件名、缩写）极常见，不能因此判成英语。"""
    assert detect_language("MATLAB 在哪里下载") == "中文"
    assert detect_language("校园网 VPN 怎么用") == "中文"


@pytest.mark.parametrize("text", ["", "   ", "123 ！？", "……"])
def test_unjudgeable_falls_back_to_default(text):
    """判不出来时退回默认语言，不抛错。"""
    assert detect_language(text) == "中文"


def test_ask_request_resolves_auto():
    """"auto" 必须被解析成具体语言，调用方拿到的永远不是 "auto"。"""
    req = AskRequest(question="How do I apply for a scholarship?", language="auto")
    assert req.language == "英语"


@pytest.mark.parametrize("sentinel", ["auto", "AUTO", " Auto ", ""])
def test_ask_request_auto_is_case_insensitive_and_tolerates_blank(sentinel):
    req = AskRequest(question="図書館の開館時間", language=sentinel)
    assert req.language == "日语"


def test_ask_request_keeps_explicit_language():
    """显式指定语言时不能被自动识别覆盖——要保留人工指定的能力。"""
    req = AskRequest(question="如何办理休学手续？", language="English")
    assert req.language == "English"


def test_ask_request_default_unchanged():
    """不传 language 时默认仍是中文，直接调 API 的老客户端行为与以往一致。"""
    assert AskRequest(question="How to apply?").language == "中文"
