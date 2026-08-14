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


class TestOrthographicVariance:
    """繁简是同一个字的两种写法，不是内容丢失。

    Whisper-zh 对同一段音频有时写简体、有时写繁体，同一次渲染里两种混着出现。
    T13 实测（帖2，12 段，同一批音频只换比较口径）：

        裸比较              89.44%
        + 数字双侧归一化    95.37%
        + 繁转简            97.59%   ← 单段最低那段从 61.76% 升到 97.06%

    音频一个字没变，升的那 2.2 个百分点全是度量假象。这跟数字归一化是同一条原则：
    正字法差异必须在比较口径里消掉，否则闸门量的是 ASR 的书写偏好而不是内容。
    """

    def test_a_traditional_transcription_of_simplified_text_is_full_coverage(self) -> None:
        assert coverage("留意波动，把控制权拿回来", "留意波動，把控制權拿回來") == 1.0

    def test_a_simplified_transcription_of_traditional_text_is_full_coverage(self) -> None:
        """两侧都要归一，不是只修听到侧——比较的是等价类，方向不该有特权。"""
        assert coverage("留意波動，把控制權拿回來", "留意波动，把控制权拿回来") == 1.0

    def test_traditional_variants_do_not_show_up_as_missing_spans(self) -> None:
        """差异报告是人判断「真丢了还是听错了」的唯一依据。繁体写法列在缺失片段里，
        人会以为内容没念出来，而实际上音频是完整的。"""
        assert missing_spans("把控制权拿回来", "把控制權拿回來") == ()

    def test_real_content_loss_still_shows_up_through_the_canonicalisation(self) -> None:
        """正控制：归一化不许把真的丢字也一起抹平。"""
        assert missing_spans("留意波动，把控制权拿回来", "留意波動") == ((4, "把控制权拿回来"),)

    def test_canonicalisation_leaves_simplified_text_alone(self) -> None:
        """简体输入必须恒等——`missing_spans` 的下标是归一化后参考串里的下标，
        而差异报告printing 的是**未**归一化的朗读文本。两者长度一旦不同，报告里的
        位置就会错位到别的字上。实测两篇真实短帖（213 / 540 汉字）皆恒等。
        """
        text = "体检数字和诊断需要认真对待，我们常顺手把一次测量翻译成结论。"
        assert missing_spans(text, text) == ()
        assert coverage(text, text) == 1.0
