"""长文切段。

第一条纪律：切段不是丢内容的许可——任何路径下汉字数都必须守恒。

第二条纪律来自一次实测失败（T13）：**一次生成里不许出现两个段落**。跨段落合并
送进模型会触发跳读，音频时长正常、听起来完整，中间少一整句。实测帖1 单段 213
汉字覆盖率 55.40%，切到段落级 99.06%。所以段落是原子单位，长度阈值只用来切开
**超长的单个段落**，不再用来把段落打包到一起。
"""

import logging

import pytest

from voxout import config
from voxout.chunker import chunk
from voxout.textmatch import han_only


def test_a_single_paragraph_under_the_threshold_is_returned_verbatim() -> None:
    """单个短段落必须原样返回——不许被重新拼装。"""
    text = "留意波动，把控制权拿回来。"
    assert chunk(text) == (text,)


def test_a_paragraph_exactly_at_the_threshold_is_not_split() -> None:
    """阈值是 `>` 而非 `>=`：刚好等于阈值仍走单次调用。"""
    text = "一二三四五六七八九十。" * 6
    assert len(han_only(text)) == config.CHUNK_THRESHOLD_HAN
    assert chunk(text) == (text,)


def test_a_paragraph_over_the_threshold_is_split() -> None:
    text = "一二三四五六七八九十。" * 7
    assert len(chunk(text)) > 1


def test_two_paragraphs_never_share_one_generation() -> None:
    """T13 的核心结论：跨段落合并会触发跳读。哪怕两段加起来远低于阈值也要分开。"""
    assert chunk("一二三。\n四五六。") == ("一二三。", "四五六。")


def test_paragraphs_stay_separate_even_when_tiny() -> None:
    """标题行只有几个字，也不许和正文并进一次生成——帖2 的 80 汉字首段就是这样丢的。"""
    assert chunk("标题\n正文一。\n正文二。") == ("标题", "正文一。", "正文二。")


def test_blank_lines_between_paragraphs_do_not_produce_empty_segments() -> None:
    """markdown 常留空行。空段落送进模型会生成一段静音。"""
    assert chunk("一二三。\n\n\n四五六。") == ("一二三。", "四五六。")


def test_han_count_is_conserved_across_segments() -> None:
    text = "一二三四五六七八九十。" * 12
    segments = chunk(text)
    assert len(han_only("".join(segments))) == len(han_only(text))


def test_han_count_is_conserved_across_paragraph_boundaries() -> None:
    text = "一二三四五六七八九十。" * 12 + "\n" + "甲乙丙丁。" * 3
    segments = chunk(text)
    assert len(han_only("".join(segments))) == len(han_only(text))


def test_greedy_packing_fills_a_segment_before_starting_the_next() -> None:
    """锁住贪心打包的确定性：句子够装就装，装不下才开新段。"""
    text = "一二三。四五六。七八九。"
    assert chunk(text, threshold_han=6) == ("一二三。四五六。", "七八九。")


def test_a_long_paragraph_is_packed_greedily_within_itself() -> None:
    """超长段落内部按句打包，但打包绝不越过段落界。"""
    text = "一二三。四五六。七八九。\n甲乙丙。"
    assert chunk(text, threshold_han=6) == ("一二三。四五六。", "七八九。", "甲乙丙。")


def test_all_terminal_punctuation_is_honoured() -> None:
    text = "一二三！四五六？七八九；"
    assert chunk(text, threshold_han=3) == ("一二三！", "四五六？", "七八九；")


def test_unsplittable_long_sentence_keeps_all_its_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """无标点超长句：可以放弃切段，不可以丢字。"""
    text = "一二三四五六七八九十" * 100  # 1000 汉字，零标点
    with caplog.at_level(logging.WARNING):
        segments = chunk(text)

    assert len(han_only("".join(segments))) == 1000
    assert any("标点" in r.message for r in caplog.records), "放弃切段必须留下告警"


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk("")
