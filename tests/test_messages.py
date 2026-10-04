"""多语言话术表自检。

这些检查看着琐碎，但都是真实踩过的坑：
- 漏补一种语言 → 用户收到一句语言不对的兜底（`_pick` 会静默退回中文）；
- 兜底话术改了、指纹没跟上 → 拒答识别失效，联网兜底不再触发。
"""

import pytest

from xtuagent.core.lang import DEFAULT_LANGUAGE, detect_language
from xtuagent.core.messages import (
    _TEXT,
    LANGUAGES,
    fallback_answer,
    service_error,
    unhandled,
    web_disclaimer,
    web_status,
)
from xtuagent.rag.pipeline import _REFUSAL_SIGNATURES, is_unanswerable


def test_languages_match_language_detector():
    """文案表覆盖的语言必须与语言识别器的产出**完全一致**。

    识别器返回了「韩语」而文案表没这一项时，用户拿到的是中文兜底——
    这类漂移不会有任何报错，只能靠这条断言挡。
    """
    detectables = {
        detect_language(t)
        for t in ("你好", "hello", "こんにちは", "안녕하세요", "привет")
    }
    assert detectables == set(LANGUAGES)


@pytest.mark.parametrize("name", sorted(_TEXT))
def test_every_message_covers_every_language(name):
    table = _TEXT[name]
    assert set(table) == set(LANGUAGES), f"{name} 的语言与 LANGUAGES 不一致"
    for lang, text in table.items():
        assert text.strip(), f"{name}[{lang}] 是空的"


def test_fallback_and_disclaimer_are_actually_translated():
    """非中文的话术不能只是把中文抄一遍——那正是用户抱怨的「割裂感」。"""
    for lang in LANGUAGES:
        if lang == DEFAULT_LANGUAGE:
            continue
        assert fallback_answer(lang) != fallback_answer(DEFAULT_LANGUAGE), lang
        assert web_disclaimer(lang) != web_disclaimer(DEFAULT_LANGUAGE), lang


def test_unknown_language_falls_back_to_chinese():
    """没收录的语言不能抛异常，退回默认即可。"""
    assert fallback_answer("世界语") == fallback_answer(DEFAULT_LANGUAGE)
    assert fallback_answer("") == fallback_answer(DEFAULT_LANGUAGE)
    assert web_status("世界语", 3) == web_status(DEFAULT_LANGUAGE, 3)


def test_web_status_interpolates_count():
    assert "5" in web_status("英语", 5)
    assert "5" in web_status("日语", 5)


def test_other_accessors_are_language_aware():
    assert service_error("英语") != service_error(DEFAULT_LANGUAGE)
    assert unhandled("日语") != unhandled(DEFAULT_LANGUAGE)


# ---------------------------------------------------------------------------
# 拒答识别：多语言覆盖
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lang", LANGUAGES)
def test_own_fallback_is_recognised_as_refusal(lang):
    """提示词要求模型照抄的那句话，必须能被自己的识别逻辑认出来。

    这一条一旦破了，外文提问的联网兜底就永远不会触发——
    表现为「明明没答上来，却也不去联网」。
    """
    assert is_unanswerable(fallback_answer(lang))


@pytest.mark.parametrize("lang", LANGUAGES)
def test_signatures_are_distinct_and_long_enough(lang):
    """指纹太短会互相撞车（把正常回答判成拒答）。"""
    from xtuagent.rag.pipeline import _compact, _SIGNATURE_LEN

    sig = _compact(fallback_answer(lang))[:_SIGNATURE_LEN]
    assert len(sig) == _SIGNATURE_LEN, f"{lang} 的兜底话术太短，指纹不足"
    assert sig in _REFUSAL_SIGNATURES


@pytest.mark.parametrize(
    "text",
    [
        # 实测中真实出现过的英文拒答（原本识别不出来）
        "According to the provided knowledge base, there is no specific information "
        "regarding the download location for MATLAB. Therefore, I cannot provide an "
        "accurate answer to this question.",
        "I am sorry, but I could not find any relevant information in the knowledge base.",
        "The knowledge base does not contain any information about this topic.",
        "I was unable to find specific information to answer your question.",
        "No information about this can be found in the knowledge base.",
        "申し訳ありませんが、知識ベースに該当する情報が見つかりません。",
        "지식 베이스에서 해당하는 정보를 찾을 수 없습니다.",
        "В текущей базе знаний не найдено соответствующей информации.",
    ],
)
def test_foreign_refusals_are_detected(text):
    assert is_unanswerable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # 含否定词但**是正常回答**——误判的代价是把好答案换成网络答案
        "You cannot submit the form late; the deadline is fixed [1].",
        "You cannot find the room without a card, so bring your student ID [1].",
        "Students cannot register for more than 30 credits per term [2].",
        "There is no exam scheduled for this course in week 10 [1].",
        "The library is open from 8:00 to 22:00 on weekdays [1].",
    ],
)
def test_english_normal_answers_are_not_mistaken_for_refusals(text):
    assert is_unanswerable(text) is False


def test_leading_markdown_does_not_hide_a_refusal():
    """模型爱把开头包成 `**加粗**`，归一化没做的话识别会整体失效。"""
    assert is_unanswerable("**根据现有知识库暂未找到相关信息**，建议咨询教务处。")
    assert is_unanswerable("> No relevant information was found in the knowledge base.")


def test_refusal_quoted_mid_text_does_not_count():
    """正文里引用一句兜底话术是正常回答，不是整条拒答。

    指纹那一路如果放开成「全文搜索」，这句就会被误判——
    代价是把一个好答案换成网络答案。判定只看开头的一小段。
    """
    text = (
        "转专业申请需在每学期开学两周内提交，逾期不再受理 [1]。"
        "学院会在第十周统一组织考核，考核内容包括笔试与面试 [2]。"
        "学生提供的材料必须真实，若根据现有知识库暂未找到相关条款，请联系教务处确认。"
    )
    assert is_unanswerable(text) is False


def test_english_refusal_with_preamble_is_detected():
    """模型常先写一句前言再照抄兜底句——前言会把拒答推到窗口深处。

    真实漏判（2026-10-02 实测）：「No relevant information」落在第 82 个字符，
    当时的 60 字符窗口扫不到，于是这条拒答被当成正常回答返回（mode=kb），
    用户只拿到一句拒答，联网兜底根本没触发。
    """
    text = (
        "The GPA warning threshold is not explicitly mentioned in the provided "
        "knowledge base. No relevant information was found in the current knowledge "
        "base. Please consult the Academic Affairs Office or your academic advisor "
        "for more details. [1]"
    )
    assert is_unanswerable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # 实测原文（2026-10-02）：「什么是摩尔定律？」得到的回答
        '很抱歉，我无法直接回答"什么是摩尔定律？"这个问题。根据我的知识库，我找到了相关信息。',
        "抱歉，我无法回答这个问题。",
        "很抱歉，我不能直接给出答案，因为知识库中没有相关内容。",
    ],
)
def test_explicit_cannot_answer_is_a_refusal(text):
    """「我无法直接回答…」必须算拒答，否则模型既没答上也不去联网。

    实测代价：摩尔定律那题检索到的全是无关内容，模型罗列一遍就收尾了，
    `is_unanswerable` 认不出 → 确定性联网兜底不触发 → 用户拿到的是一堆没用的片段。
    """
    assert is_unanswerable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "我无法确认该信息的准确性，但根据知识库 [1]，转专业需提交申请表。",
        "如无法按时提交，请提前联系教务处说明情况 [2]。",
        "学生不能同时申请两个专业 [1]。",
    ],
)
def test_mere_uncertainty_is_not_a_refusal(text):
    """正文里出现「无法/不能」是正常表述，宾语不是「回答/答案」就不算拒答。"""
    assert is_unanswerable(text) is False
