"""数字与单位归一化。

策略是「消除误读风险」而不是「检测误读」：把阿拉伯数字提前变成确定的中文读法，
模型就没有自由度读错。每个用例的期望值都对着 cn2an 的实测输出写，不是猜的。
"""

import logging

import pytest

from voxout import normalize
from voxout.normalize import UNIT_READINGS, to_speakable_numbers


class TestPreRulesFixCn2anFailures:
    """cn2an 裸输出会错的四种情况（计划 §Task 4 表格前四行）。"""

    def test_four_digit_year_is_read_digit_by_digit(self) -> None:
        # 裸 cn2an：二千零二十六年
        assert to_speakable_numbers("2026 年 8 月 13 日") == "二零二六年八月十三日"

    def test_thousands_separator_is_removed_before_conversion(self) -> None:
        # 裸 cn2an：一,二百元
        assert to_speakable_numbers("1,200 元") == "一千二百元"

    def test_numeric_range_becomes_a_spoken_range(self) -> None:
        # 裸 cn2an：第三负五章——连字符被当成负号
        assert to_speakable_numbers("第 3-5 章") == "第三到五章"

    def test_leading_minus_moves_outside_the_percent(self) -> None:
        # 裸 cn2an：百分之负五
        assert to_speakable_numbers("下降了 -5%") == "下降了负百分之五"

    def test_unit_symbol_expands_to_chinese(self) -> None:
        assert to_speakable_numbers("采样率 48kHz") == "采样率四十八千赫兹"


class TestAlreadyCorrectCasesAreLocked:
    """cn2an 本来就对的情况——写成测试锁住现状，防止前置规则回归时打坏它们。"""

    def test_ordinal_day_from_the_real_corpus(self) -> None:
        assert to_speakable_numbers("第 122 天") == "第一百二十二天"

    def test_date_without_year(self) -> None:
        assert to_speakable_numbers("7 月 24 号") == "七月二十四号"

    def test_decimal_percentage(self) -> None:
        assert to_speakable_numbers("72.2%") == "百分之七十二点二"


def test_latin_words_keep_their_spacing() -> None:
    """空白折叠只发生在两个中日韩字符之间，拉丁词不能被粘住。"""
    assert to_speakable_numbers("M4 Pro 有 12 核") == "M四 Pro 有十二核"


def test_longer_unit_keys_win_over_shorter_prefixes() -> None:
    """若按字典序替换，Hz 会先吃掉 kHz 里的 Hz，得到「48k赫兹」。"""
    assert "千赫兹" in to_speakable_numbers("48kHz")
    assert "k赫兹" not in to_speakable_numbers("48kHz")


def test_unit_readings_is_a_plain_editable_mapping() -> None:
    """用户要能直接改读法（设计 §11 的人工听审待办）。"""
    assert UNIT_READINGS["kHz"] == "千赫兹"
    assert UNIT_READINGS["GB"] == "吉字节"


def test_text_without_digits_is_returned_unchanged() -> None:
    text = "留意波动，把控制权拿回来。"
    assert to_speakable_numbers(text) == text


def test_empty_text_is_returned_unchanged() -> None:
    assert to_speakable_numbers("") == ""


def test_conversion_failure_falls_back_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """cn2an 抛异常时不能静默吞掉，也不能让整次渲染挂掉。

    数字读成阿拉伯原文只是不自然；静默失败会让人以为归一化生效了。
    """

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("模拟 cn2an 内部失败")

    monkeypatch.setattr(normalize.cn2an, "transform", boom)
    with caplog.at_level(logging.WARNING):
        result = to_speakable_numbers("第 122 天")

    assert "122" in result, "回退必须保留原始内容，不能丢字"
    assert any("cn2an" in r.message for r in caplog.records), "失败必须留下告警"
