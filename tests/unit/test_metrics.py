"""度量原语，全部对闭式真值校准。

期望值不是猜的：每一个都先用正弦波数值实算过。正弦波的基频
和频谱是已知的，所以估计器出错时能立刻分辨是「估计器错」还是「音频变了」。
"""

import numpy as np
import pytest

from voxout import config
from voxout.metrics import (
    cosine,
    frames_of,
    ltas,
    median_f0,
    p10_f0,
    spread_pct,
    voiced_f0s,
)

SR = config.OUTPUT_SAMPLE_RATE


def sine(freq: float, seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestFrames:
    def test_frame_count_follows_the_hop(self) -> None:
        signal = sine(200.0)  # 96000 样本
        expected = 1 + (96000 - config.FRAME) // config.HOP
        assert frames_of(signal).shape == (expected, config.FRAME)

    def test_stereo_or_batched_input_is_refused(self) -> None:
        """二维输入会被当成「一帧很长的信号」而静默算出一个无意义的 F0。

        度量是闸门 2 的唯一信息源，在这里明确失败比往下传一个数安全。
        """
        with pytest.raises(ValueError, match="单声道"):
            frames_of(np.zeros((2, 4800), dtype=np.float32))

    def test_signal_shorter_than_one_frame_still_yields_one_frame(self) -> None:
        assert frames_of(sine(200.0, seconds=0.01)).shape[0] == 1


class TestF0AgainstClosedFormTruth:
    def test_two_hundred_hertz_sine_is_estimated_within_one_percent(self) -> None:
        signal = sine(200.0)
        assert p10_f0(signal, SR) == pytest.approx(200.0, rel=0.01)
        assert median_f0(signal, SR) == pytest.approx(200.0, rel=0.01)

    def test_octave_error_above_the_search_range_is_locked_in(self) -> None:
        """440 Hz 被判成 220.2 Hz——这是取值的后果，不是缺陷。

        真周期 109 样本 < lag_min = sr/F0_MAX_HZ = 120，第一个峰被排除在搜索窗外，
        argmax 落在 2 倍周期（218 样本）上。断言写死，防止日后有人把 F0_MAX_HZ
        当 bug「修」大——那会让人声的谐波峰重新进入搜索窗，代价远大于收益。
        """
        assert 215.0 < p10_f0(sine(440.0), SR) < 226.0

    def test_silence_returns_nan_rather_than_crashing(self) -> None:
        """全静音下自相关的零延迟项为 0，除法必须被短路而不是抛异常。"""
        assert np.isnan(p10_f0(np.zeros(96000, dtype=np.float32), SR))
        assert np.isnan(median_f0(np.zeros(96000, dtype=np.float32), SR))

    def test_sample_rate_is_explicit_not_assumed(self) -> None:
        """44.1kHz 的锚点若被按 48kHz 解读，F0 会偏 8.8%——刚好啃掉闸门 2 的余量。"""
        assert p10_f0(sine(200.0, sr=44100), 44100) == pytest.approx(200.0, rel=0.01)

    def test_voiced_f0s_returns_one_estimate_per_voiced_frame(self) -> None:
        assert len(voiced_f0s(sine(200.0), SR)) == frames_of(sine(200.0)).shape[0]


class TestP10DiscriminatesWhereMedianDoesNot:
    """设计 §4 的方法论陷阱的量化形式。

    中位数被语调污染：同一个人念不同句子，中位音高天然有 5.1% 极差，
    已经吃掉了闸门 2 的大半余量。p10（浊音音高地板）几乎不动。
    """

    def test_median_moves_with_intonation_while_p10_holds_the_floor(self) -> None:
        # 前 2 秒 200 Hz、后 2 秒 250 Hz：模拟同一说话人抬高了语调。
        mixed = np.concatenate([sine(200.0), sine(250.0)])
        assert p10_f0(mixed, SR) == pytest.approx(200.0, rel=0.01)
        assert median_f0(mixed, SR) > 210.0


class TestLtasAndCosine:
    def test_identical_spectra_correlate_perfectly(self) -> None:
        spectrum = ltas(sine(200.0), SR)
        assert cosine(spectrum, spectrum) == pytest.approx(1.0)

    def test_far_apart_tones_have_low_correlation(self) -> None:
        assert cosine(ltas(sine(200.0), SR), ltas(sine(2000.0), SR)) < 0.5

    def test_nearby_tones_correlate_highly_which_is_why_p10_leads(self) -> None:
        """200 Hz vs 250 Hz 的 LTAS 余弦实测 0.990——25% 的音高差它几乎看不见。

        这条测试记录的是 LTAS 的分辨力上限：它是音色（频谱包络）的指纹，
        不是音高的指纹。所以闸门 2 以 p10 F0 为判据，LTAS 只作旁证。
        """
        assert cosine(ltas(sine(200.0), SR), ltas(sine(250.0), SR)) > 0.9

    def test_a_constant_spectrum_yields_nan_not_perfect_similarity(self) -> None:
        """去均值后方差为 0，相关性在此处无定义。

        这个分支的存在理由是它的**替代品有多坏**：返回 1.0 等于告诉闸门
        「两段音色完全一致」，而输入其实是一段没有任何频谱结构的退化信号
        （全静音、直流）。断言 nan 是为了把这条挡住的路钉住。
        """
        flat = np.ones(64, dtype=np.float32)
        assert np.isnan(cosine(flat, flat))
        assert np.isnan(cosine(flat, np.arange(64, dtype=np.float32)))

    def test_spectrum_is_truncated_to_the_configured_ceiling(self) -> None:
        expected_bins = int(config.LTAS_MAX_HZ / (SR / config.FRAME))
        assert len(ltas(sine(200.0), SR)) == expected_bins


class TestSpread:
    def test_identical_values_have_zero_spread(self) -> None:
        assert spread_pct([100.0, 100.0, 100.0]) == 0.0

    def test_spread_is_range_over_mean(self) -> None:
        """分母是均值，与设计 §4 的标尺（0.9% / 3.4% / 53.9%）保持同一定义。"""
        assert spread_pct([100.0, 110.0]) == pytest.approx(100.0 * 10.0 / 105.0)

    def test_single_value_has_zero_spread(self) -> None:
        assert spread_pct([180.0]) == 0.0

    def test_nan_values_are_ignored(self) -> None:
        assert spread_pct([100.0, float("nan"), 110.0]) == pytest.approx(
            100.0 * 10.0 / 105.0
        )

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            spread_pct([])

    def test_all_nan_input_is_rejected(self) -> None:
        """全 nan 意味着没有一帧浊音——静默返回 0% 会让闸门 2 假装通过。"""
        with pytest.raises(ValueError):
            spread_pct([float("nan"), float("nan")])
