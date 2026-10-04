"""提问语言识别：让系统用提问者自己的语言作答。

为什么不引第三方语言检测库：
本机装包代价高（见项目备忘里的网络限制），而这里只需要在
「中日韩 / 西里尔 / 拉丁」之间做个粗判——**字符集判定足够可靠**，
且完全离线、零依赖、零开销。

为什么顺序是「假名 → 谚文 → 汉字」：
日文句子必然夹假名，而汉字区间会**同时命中中日文**。
若先判汉字，日文提问会被误判成中文。
"""

import re

# 假名（平假名 U+3040-309F + 片假名 U+30A0-30FF）
_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
# 谚文（韩文音节 + 字母）
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
# 汉字（CJK 统一表意文字 + 扩展 A）
_HAN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# 西里尔字母
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
# 拉丁字母
_LATIN = re.compile(r"[A-Za-z]")

# 判不出来时的兜底：与接口原有默认值保持一致，行为可预期。
DEFAULT_LANGUAGE = "中文"


def detect_language(text: str) -> str:
    """按字符集粗判提问语言，返回可直接写进提示词的语言名。

    返回值用中文命名（中文/英语/日语/韩语/俄语），因为系统提示词本身是中文，
    「回答使用日语语言」比「回答使用日本語语言」读起来更连贯。
    """
    if not text:
        return DEFAULT_LANGUAGE
    if _KANA.search(text):
        return "日语"
    if _HANGUL.search(text):
        return "韩语"
    if _HAN.search(text):
        return "中文"
    if _CYRILLIC.search(text):
        return "俄语"
    if _LATIN.search(text):
        return "英语"
    return DEFAULT_LANGUAGE
