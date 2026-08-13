"""汉字覆盖率。

这个模块唯一的难点是「局部替换不得引起永久失步」——设计 §6 记录的那个 bug。
手写贪心双指针在遇到一处不匹配后会一路错位，把 90% 的相似度读成 3.5%。
"""

from voxout.textmatch import coverage, han_only, missing_spans


def test_han_only_keeps_only_han() -> None:
    assert han_only("第 122 天，读了 3 页！") == "第天读了页"


def test_identical_text_is_full_coverage() -> None:
    assert coverage("一二三四五", "一二三四五") == 1.0


def test_missing_tail_halves_the_coverage() -> None:
    assert coverage("一二三四五六七八九十", "一二三四五") == 0.5


def test_local_substitution_does_not_desync_the_rest() -> None:
    """回归锁（设计 §6）。

    合成的是「第一百二十二天」，Whisper 写回「第122天」。`han_only` 后参考侧
    11 字、听到侧 10 字，只差一个「三」。稳健的比较应给出 ~0.91；
    手写贪心版在此崩到 3.5%，那正是这条测试要永久挡住的东西。
    """
    got = coverage("第一百二十二天读了三页", "第一百二十二天读了3页")
    assert got >= 0.9, f"局部替换引起了失步：coverage={got:.3f}"


def test_empty_reference_is_zero_not_a_crash() -> None:
    assert coverage("", "随便") == 0.0


def test_empty_heard_is_zero() -> None:
    assert coverage("一二三", "") == 0.0


def test_missing_spans_reports_offset_and_text() -> None:
    assert missing_spans("一二三四五六", "一二五六") == ((2, "三四"),)


def test_missing_spans_is_empty_when_nothing_is_missing() -> None:
    assert missing_spans("一二三", "一二三") == ()


def test_missing_spans_ignores_non_han_noise_in_heard() -> None:
    """听到侧多出的标点不算缺失——缺失只针对参考侧的字。"""
    assert missing_spans("一二三", "一，二。三！") == ()
