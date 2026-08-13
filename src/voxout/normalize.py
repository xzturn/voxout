"""数字与单位归一化：把误读风险**消除**，而不是事后检测。

思路：模型读「122」时有自由度（一二二 / 一百二十二 / 幺二二），读「一百二十二」
时没有。所以在送进模型之前就把数字定成中文读法，闸门 1 只需管真正的丢字。

流水线顺序由 cn2an 的实测失败模式决定，不能重排：

    千分位 → 年份 → 区间 → 负号百分比 → 单位 → cn2an → 折叠 CJK 间空白

最后一步不是美化。cn2an 会原样保留空格（`1200 元` → `一千二百 元`），
而 TTS 会把中文里的空格读成停顿。
"""

import logging
import re

import cn2an

_LOG = logging.getLogger(__name__)

UNIT_READINGS: dict[str, str] = {
    "kHz": "千赫兹",
    "Hz": "赫兹",
    "GB": "吉字节",
    "MB": "兆字节",
    "KB": "千字节",
    "TB": "太字节",
    "ms": "毫秒",
    "km": "公里",
    "cm": "厘米",
    "kg": "公斤",
    "°C": "摄氏度",
}
"""单位符号的中文读法。用户可直接编辑；自然度需人工听审确认（设计 §11）。"""

_DIGIT_READINGS = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}  # fmt: skip

_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d)")
_YEAR = re.compile(r"(\d{4})\s*年")
_NUMERIC_RANGE = re.compile(r"(\d)\s*[-–—~～]\s*(?=\d)")
_NEGATIVE_PERCENT = re.compile(r"(?<!\d)-\s*(\d+(?:\.\d+)?)%")
_CJK = r"　-〿一-鿿＀-￯"
_CJK_INNER_SPACE = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")


def _read_year_digit_by_digit(match: re.Match[str]) -> str:
    """2026 年 → 二零二六年。cn2an 会给出「二千零二十六年」，是数量而非年份的读法。"""
    return "".join(_DIGIT_READINGS[d] for d in match.group(1)) + "年"


def _expand_units(text: str) -> str:
    """按 key 长度降序替换：否则 `Hz` 先命中 `kHz` 的后半，留下「48k赫兹」。"""
    for symbol in sorted(UNIT_READINGS, key=len, reverse=True):
        text = text.replace(symbol, UNIT_READINGS[symbol])
    return text


def _apply_pre_rules(text: str) -> str:
    text = _THOUSANDS_COMMA.sub("", text)
    text = _YEAR.sub(_read_year_digit_by_digit, text)
    # 连字符不先换掉，cn2an 会把它当负号：第 3-5 章 → 第三负五章。
    text = _NUMERIC_RANGE.sub(r"\1 到 ", text)
    # 负号必须紧贴数字，`负 5%` 会被转成「负 百分之五」，负号脱离了百分之。
    text = _NEGATIVE_PERCENT.sub(r"负\1%", text)
    return _expand_units(text)


def _collapse_cjk_spacing(text: str) -> str:
    """删掉两个中日韩字符之间的空格，保留拉丁词两侧的空格与所有换行。

    换行是 chunker 的切段点，绝不能折叠。
    """
    return _REPEATED_SPACE.sub(" ", _CJK_INNER_SPACE.sub("", text))


def to_speakable_numbers(text: str) -> str:
    """把阿拉伯数字与单位符号转成确定的中文读法。

    cn2an 失败时保留原文并告警——数字读成阿拉伯原文只是不自然，
    而静默失败会让人误以为归一化生效了。
    """
    if not text:
        return text

    prepared = _apply_pre_rules(text)
    try:
        converted = cn2an.transform(prepared, "an2cn")
    except Exception as exc:  # cn2an 对畸形输入的异常类型未文档化，只能宽捕获
        _LOG.warning("cn2an 数字转换失败（%s），保留阿拉伯数字原文", exc)
        converted = prepared

    return _collapse_cjk_spacing(converted)
