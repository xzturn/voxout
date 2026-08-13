"""长文切段。

切段的唯一目的是绕开 token 上限。因此第一条纪律是：切段不是丢内容的许可——
任何路径下汉字数都必须守恒。
"""

import logging

import pytest

from voxout import config
from voxout.chunker import chunk
from voxout.textmatch import han_only


def test_short_text_is_returned_verbatim_as_one_segment() -> None:
    """真实短帖 283 汉字，远低于阈值，必须原样单段——不许被重新拼装。"""
    text = "留意波动，把控制权拿回来。" * 10
    assert chunk(text) == (text,)


def test_text_exactly_at_the_threshold_is_not_split() -> None:
    """阈值是 `>` 而非 `>=`：刚好 900 字仍走单次调用。"""
    text = "一二三四五六七八九十。" * 90
    assert len(han_only(text)) == config.CHUNK_THRESHOLD_HAN
    assert chunk(text) == (text,)


def test_text_over_the_threshold_is_split() -> None:
    text = "一二三四五六七八九十。" * 91
    segments = chunk(text)
    assert len(segments) > 1


def test_han_count_is_conserved_across_segments() -> None:
    text = "一二三四五六七八九十。" * 120
    segments = chunk(text)
    assert len(han_only("".join(segments))) == len(han_only(text))


def test_greedy_packing_fills_a_segment_before_starting_the_next() -> None:
    """锁住贪心打包的确定性：句子够装就装，装不下才开新段。"""
    text = "一二三。四五六。七八九。"
    assert chunk(text, threshold_han=6) == ("一二三。四五六。", "七八九。")


def test_newline_is_a_split_point() -> None:
    """markdown 清洗后段落界是 \\n，切段要认它。"""
    assert chunk("一二三\n四五六\n七八九", threshold_han=6) == ("一二三\n四五六", "七八九")


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
