"""面向用户的固定话术（多语言）。

为什么不把这几句话留在 `rag/pipeline.py` 里当常量：

1. 它们会**出现在回答正文中间**（兜底答复、联网免责声明、联网状态提示、
   服务异常提示），而回答语言由提问语言决定（见 `core.lang.detect_language`）。
   一句中文兜底贴在英文回答尾巴上，用户看到的是自相矛盾的两截。
2. `is_unanswerable()` 要靠「我们让模型照抄的那句话」来认拒答，
   提示词与识别逻辑必须**共用同一份文案**——
   否则改了文案忘了改规则，拒答识别会静默失效（英文拒答曾因此漏判）。

结构：`_TEXT[话术名][语言]`。新增语言时每个话术各补一条，
`tests/test_messages.py` 会挡住漏网之鱼。
"""

from .lang import DEFAULT_LANGUAGE

# ---------------------------------------------------------------------------
# 话术表
# ---------------------------------------------------------------------------

_TEXT = {
    # 知识库与网络都没答案时的兜底答复。
    # 同时被写进提示词，要求模型原样照抄 —— 这是拒答识别最可靠的一路信号。
    "fallback": {
        "中文": "根据现有知识库暂未找到相关信息，建议咨询教务处或辅导员获取更详细的说明。",
        "英语": (
            "No relevant information was found in the current knowledge base. "
            "Please consult the Academic Affairs Office or your academic advisor "
            "for more details."
        ),
        "日语": (
            "現在のナレッジベースには該当する情報が見つかりませんでした。"
            "詳細は教務課または指導教員にお問い合わせください。"
        ),
        "韩语": (
            "현재 지식 베이스에서 관련 정보를 찾지 못했습니다. "
            "자세한 내용은 교무처나 지도교수에게 문의하시기 바랍니다."
        ),
        "俄语": (
            "В текущей базе знаний не найдено соответствующей информации. "
            "За подробностями обратитесь в учебный отдел или к своему куратору."
        ),
    },
    # 联网回答的免责声明。由服务端强制注入，不交给模型（实测它会漏）。
    "disclaimer": {
        "中文": "**以下内容来自网络检索，非学校官方发布，请以学校官方通知为准。**\n\n",
        "英语": (
            "**The following content comes from web search results and is not "
            "officially published by the university. Please refer to the "
            "university's official announcements.**\n\n"
        ),
        "日语": (
            "**以下の内容はウェブ検索によるものであり、大学が公式に発表したものでは"
            "ありません。大学の公式なお知らせをご確認ください。**\n\n"
        ),
        "韩语": (
            "**다음 내용은 웹 검색 결과로, 학교가 공식적으로 발표한 내용이 아닙니다. "
            "학교의 공식 공지를 확인해 주세요.**\n\n"
        ),
        "俄语": (
            "**Следующая информация получена из результатов веб-поиска и не является "
            "официальной публикацией университета. Пожалуйста, ориентируйтесь на "
            "официальные объявления университета.**\n\n"
        ),
    },
    # 流式联网时的过渡提示。{count} 为检索到的条数。
    "web_status": {
        "中文": "校内知识库未收录，已联网检索到 {count} 条资料",
        "英语": "Not covered by the campus knowledge base — found {count} source(s) online.",
        "日语": "学内ナレッジベースに未収録のため、ウェブで {count} 件の資料を検索しました。",
        "韩语": "교내 지식 베이스에 없어 웹에서 자료 {count}건을 검색했습니다.",
        "俄语": "В базе знаний кампуса информации нет — найдено {count} материалов в интернете.",
    },
    # Agent 模式整轮失败。
    "service_error": {
        "中文": "抱歉，本次查询未能完成，请稍后重试或改用普通问答模式。",
        "英语": (
            "Sorry, this request could not be completed. Please try again later "
            "or switch to the standard Q&A mode."
        ),
        "日语": "申し訳ありませんが、今回の問い合わせを完了できませんでした。後でもう一度お試しいただくか、通常のQ&Aモードをご利用ください。",
        "韩语": "죄송합니다. 이번 요청을 완료하지 못했습니다. 잠시 후 다시 시도하시거나 일반 질의응답 모드를 이용해 주세요.",
        "俄语": "Извините, не удалось выполнить этот запрос. Повторите попытку позже или воспользуйтесь обычным режимом вопросов и ответов.",
    },
    # Agent 没有产出任何可用内容。
    "unhandled": {
        "中文": "无法处理该请求。",
        "英语": "Unable to handle this request.",
        "日语": "このリクエストを処理できませんでした。",
        "韩语": "이 요청을 처리할 수 없습니다.",
        "俄语": "Не удалось обработать этот запрос.",
    },
}

LANGUAGES = tuple(_TEXT["fallback"])


def _pick(name: str, language: str) -> str:
    """取话术；语言缺失时退回默认（中文），保证永不抛异常。

    宁可回一句语言不对的兜底，也不能因为没收录某种语言就让整个问答 500。
    """
    table = _TEXT[name]
    return table.get(language) or table[DEFAULT_LANGUAGE]


def fallback_answer(language: str = DEFAULT_LANGUAGE) -> str:
    """知识库与网络都没答案时的兜底话术。"""
    return _pick("fallback", language)


def web_disclaimer(language: str = DEFAULT_LANGUAGE) -> str:
    """联网回答的免责声明（服务端强制注入，不交给模型）。"""
    return _pick("disclaimer", language)


def web_status(language: str, count: int) -> str:
    """联网检索完成后的过渡提示。"""
    return _pick("web_status", language).format(count=count)


def service_error(language: str = DEFAULT_LANGUAGE) -> str:
    """Agent 整轮调用失败时的提示。"""
    return _pick("service_error", language)


def unhandled(language: str = DEFAULT_LANGUAGE) -> str:
    """Agent 没产出任何内容时的占位答复。"""
    return _pick("unhandled", language)
