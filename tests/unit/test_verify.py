"""两道闸门。全部用假 transcriber 与合成正弦，不碰任何模型。

闸门存在的理由是「静默失效」：丢字和换声都不会报错，只会让成品悄悄变坏。
所以这些测试重点盯两件事——阈值判定正确，以及**判定本身可复现**。
"""

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

from voxout import normalize, wavio
from voxout.config import COVERAGE_MIN, OUTPUT_SAMPLE_RATE, VOICE_DRIFT_MAX_PCT
from voxout.verify import ContentGate, VoiceGate, check_content, check_voice

THREE_SENTENCES = "第一段内容。第二段内容。第三段内容。"

# 自相关峰值落在整数延迟格上，Hann 窗又让自相关随延迟衰减，峰值被拉向短一格：
# 200 Hz 的理论延迟 240 → 实取 239 → 48000/239 = 200.84。偏差上界约 1/延迟
# （本频段 0.4–0.6%），是系统性的，而真实基线由同一函数在真实音频上测得，
# 所以在闸门里同偏同消。这里用实测值而非标称值，才能把「零漂移」测成真正的零。
P10_OF_200HZ = 48000 / 239
P10_OF_250HZ = 48000 / 191


def sine(freq: float = 200.0, seconds: float = 2.0, sr: int = OUTPUT_SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@dataclass(frozen=True)
class FakeTranscription:
    text: str


class RecordingTranscriber:
    """替身：回一段固定文本，并记录自己被怎么调用的。

    记录调用参数不是多余的——闸门的可复现性完全取决于 language 与 temperature
    有没有被显式传进去，那是行为的一部分，必须被测试盯住。
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[tuple[str, dict[str, object]]] = []

    def generate(self, audio: str, **kwargs: object) -> FakeTranscription:
        self.calls.append((audio, kwargs))
        return FakeTranscription(text=self._text)


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "rendered.wav"
    wavio.write(path, sine(), OUTPUT_SAMPLE_RATE)
    return path


class TestContentGate:
    def test_faithful_transcription_reaches_full_coverage(self, wav: Path) -> None:
        gate = check_content(wav, THREE_SENTENCES, RecordingTranscriber(THREE_SENTENCES))

        assert gate.coverage == 1.0
        assert gate.passed
        assert gate.missing == ()
        assert gate.threshold == COVERAGE_MIN

    def test_truncated_transcription_fails_and_locates_the_loss(self, wav: Path) -> None:
        """静默截断是选型阶段 Kokoro 的失效模式：丢 27% 内容而不报错。

        闸门不只判失败，还要指出丢在哪——否则人只知道「不对」，不知道去看哪一段。
        """
        heard = "第一段内容。第二段内容。"

        gate = check_content(wav, THREE_SENTENCES, RecordingTranscriber(heard))

        assert gate.coverage == pytest.approx(10 / 15)
        assert not gate.passed
        assert gate.missing == ((10, "第三段内容"),)

    def test_numeral_form_difference_is_not_counted_as_missing_content(
        self, wav: Path
    ) -> None:
        """第四条修正的锁。

        朗读文本已被归一化成「第一百二十二天」，而 ASR 按听到的音回写成
        「第122天」。两者内容完全一致，但字面只共享「第」「天」两字——
        单侧比较得 0.2857（实测），会把一次完美的渲染判成丢了 71% 的内容。
        双侧都过归一化后是 1.0。
        """
        gate = check_content(wav, "第一百二十二天", RecordingTranscriber("第122天"))

        assert gate.coverage == 1.0
        assert gate.passed

    def test_asr_is_called_with_explicit_language_and_zero_temperature(
        self, wav: Path
    ) -> None:
        """whisper 的 temperature 默认是回退元组 (0.0, 0.2, …, 1.0)。

        一旦触发回退，同一段音频两次转写可能给出不同数字，闸门就不再可复现。
        语言同理：让它自动检测，中文短句偶尔会被判成日文。
        """
        transcriber = RecordingTranscriber(THREE_SENTENCES)

        check_content(wav, THREE_SENTENCES, transcriber)

        assert transcriber.calls == [(str(wav), {"language": "zh", "temperature": 0.0})]

    def test_raw_asr_text_is_preserved_for_diagnosis(self, wav: Path) -> None:
        """报告里要留 ASR 的原话，而不是归一化后的版本——人排查时看的是原话。"""
        gate = check_content(wav, "第一百二十二天", RecordingTranscriber("第122天"))

        assert gate.asr_text == "第122天"

    def test_normalization_failure_is_flagged_but_comparison_still_runs(
        self, wav: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """归一化失败不该让闸门瘫掉，但必须留痕。

        否则覆盖率因工具故障而偏低时，人会误判成内容丢失，去改一个没坏的东西。
        """

        def boom(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("模拟 cn2an 故障")

        monkeypatch.setattr(normalize.cn2an, "transform", boom)

        gate = check_content(wav, THREE_SENTENCES, RecordingTranscriber(THREE_SENTENCES))

        assert gate.asr_normalize_failed
        assert gate.coverage == 1.0
        assert gate.passed

    def test_missing_audio_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope.wav"):
            check_content(tmp_path / "nope.wav", THREE_SENTENCES, RecordingTranscriber(""))

    def test_empty_spoken_text_is_refused(self, wav: Path) -> None:
        with pytest.raises(ValueError, match="为空"):
            check_content(wav, "  \n", RecordingTranscriber(""))

    def test_content_gate_is_immutable(self, wav: Path) -> None:
        gate = check_content(wav, THREE_SENTENCES, RecordingTranscriber(THREE_SENTENCES))

        with pytest.raises(FrozenInstanceError):
            gate.passed = False  # type: ignore[misc]


class TestVoiceGate:
    def test_segment_reproducing_the_baseline_has_exactly_no_drift(self) -> None:
        """同一段音频对自己测出的基线，漂移必须是精确的 0——不是「接近 0」。

        这条钉住了「基线代表可复现的那个人」这个前提：若同一信号对自身基线
        都有漂移，闸门的 6% 预算就有一部分被估计噪声吃掉了。
        """
        gate = check_voice([sine(200.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        assert gate.drift_pct == pytest.approx(0.0, abs=1e-9)
        assert gate.passed
        assert gate.threshold == VOICE_DRIFT_MAX_PCT

    def test_segment_far_from_the_baseline_fails(self) -> None:
        """250 Hz 对 200 Hz 基线 ≈ 25% 漂移，远超 6% 余量：这是换了个人说话。"""
        gate = check_voice([sine(250.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        expected = 100.0 * (P10_OF_250HZ - P10_OF_200HZ) / P10_OF_200HZ
        assert gate.drift_pct == pytest.approx(expected)
        assert gate.drift_pct == pytest.approx(25.1, abs=0.1)
        assert not gate.passed

    def test_single_segment_has_no_inter_segment_spread(self) -> None:
        gate = check_voice([sine(200.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        assert gate.spread_pct == 0.0
        assert len(gate.segment_p10s) == 1

    def test_multiple_segments_report_their_spread(self) -> None:
        """段间极差是旁证：判定用的是对基线的漂移，但极差能指出是哪一段跑了。"""
        gate = check_voice(
            [sine(200.0), sine(250.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ
        )

        # 分母是均值，与设计 §4 的标尺同定义：100 × 极差 / 均值 = 22.3%
        mean = (P10_OF_200HZ + P10_OF_250HZ) / 2
        assert gate.spread_pct == pytest.approx(
            100.0 * (P10_OF_250HZ - P10_OF_200HZ) / mean
        )
        assert gate.segment_p10s == pytest.approx((P10_OF_200HZ, P10_OF_250HZ))

    def test_silent_segment_fails_instead_of_silently_passing(self) -> None:
        """全静音测不出 p10 F0。nan 必须传播成失败，而不是被当作「零漂移」。"""
        gate = check_voice(
            [np.zeros(OUTPUT_SAMPLE_RATE, dtype=np.float32)],
            OUTPUT_SAMPLE_RATE,
            baseline_p10=200.0,
        )

        assert not gate.passed
        assert np.isnan(gate.drift_pct)

    def test_one_silent_segment_among_good_ones_still_fails(self) -> None:
        gate = check_voice(
            [sine(200.0), np.zeros(OUTPUT_SAMPLE_RATE, dtype=np.float32)],
            OUTPUT_SAMPLE_RATE,
            baseline_p10=200.0,
        )

        assert not gate.passed

    def test_non_positive_baseline_is_refused(self) -> None:
        with pytest.raises(ValueError, match="基线"):
            check_voice([sine()], OUTPUT_SAMPLE_RATE, baseline_p10=0.0)

    def test_no_segments_is_refused(self) -> None:
        with pytest.raises(ValueError, match="没有"):
            check_voice([], OUTPUT_SAMPLE_RATE, baseline_p10=200.0)

    def test_voice_gate_is_immutable(self) -> None:
        gate = check_voice([sine()], OUTPUT_SAMPLE_RATE, baseline_p10=200.0)

        with pytest.raises(FrozenInstanceError):
            gate.drift_pct = 0.0  # type: ignore[misc]


def test_gates_are_reported_not_raised() -> None:
    """闸门失败返回 passed=False，不抛异常。

    渲染产物仍要落盘：人需要听一下才能判断是模型的问题还是 ASR 的问题。
    把失败变成异常会让音频丢失，反而更难排查。
    """
    gate = check_voice([sine(400.0)], OUTPUT_SAMPLE_RATE, baseline_p10=200.0)

    assert isinstance(gate, VoiceGate)
    assert not gate.passed


def test_content_gate_exposes_all_documented_fields() -> None:
    gate = ContentGate(
        coverage=1.0,
        threshold=0.97,
        passed=True,
        asr_text="文本",
        missing=(),
        asr_normalize_failed=False,
    )
    assert gate.asr_normalize_failed is False
