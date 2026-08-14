"""两道闸门。全部用假 transcriber 与合成正弦，不碰任何模型。

闸门存在的理由是「静默失效」：丢字和换声都不会报错，只会让成品悄悄变坏。
所以这些测试重点盯两件事——阈值判定正确，以及**判定本身可复现**。
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from fakes import P10_OF_200HZ, P10_OF_250HZ, RecordingTranscriber, sine

from voxout import normalize, wavio
from voxout.config import COVERAGE_MIN, OUTPUT_SAMPLE_RATE, VOICE_DRIFT_MAX_PCT
from voxout.verify import ContentGate, VoiceGate, check_content, check_voice

THREE_SENTENCES = "第一段内容。第二段内容。第三段内容。"


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
        都有漂移，闸门的余量就有一部分被估计噪声吃掉了。
        """
        gate = check_voice([sine(200.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        assert gate.drift_pct == pytest.approx(0.0, abs=1e-9)
        assert gate.passed
        assert gate.threshold == VOICE_DRIFT_MAX_PCT

    def test_segment_far_from_the_baseline_fails(self) -> None:
        """250 Hz 对 200 Hz 基线 ≈ 25% 漂移，远超 15% 上限：这是换了个人说话。"""
        gate = check_voice([sine(250.0)], OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        expected = 100.0 * (P10_OF_250HZ - P10_OF_200HZ) / P10_OF_200HZ
        assert gate.drift_pct == pytest.approx(expected)
        assert gate.drift_pct == pytest.approx(25.1, abs=0.1)
        assert not gate.passed

    def test_one_deviant_segment_among_many_does_not_decide_the_verdict(self) -> None:
        """判定看多数段，不看最跑偏的那一段。

        换掉 `np.max` 的理由是实测的：真实内容里单段偏离十几个百分点是常态，
        cal2 的第 11 段 +12.6%、第 13 段 −13.8%，而全篇音色由人听是一致的。
        极值统计量还会随段数长——6 段的 max 是 10.65%，14 段是 13.79%，同一批
        音频的中位数则是 5.35% 与 4.72%。段数越多判得越严，不是闸门该有的性质。

        离群段并不因此消失：它照旧留在 `segment_p10s` 与 `spread_pct` 里给人看。
        """
        segments = [sine(200.0)] * 5 + [sine(250.0)]

        gate = check_voice(segments, OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        assert gate.drift_pct == pytest.approx(0.0, abs=1e-9)
        assert gate.passed
        assert gate.spread_pct > 20.0, "离群段必须仍然作为旁证可见"

    def test_a_majority_of_deviant_segments_still_fails(self) -> None:
        """多数段跑偏才是「换了个人」，这个必须照旧挡住。

        与上一条只差一个段的归属（2:3 而非 5:1），两条合起来钉住判据是**比例**
        而不是极值——只有中位数同时满足这两条。
        """
        segments = [sine(200.0)] * 2 + [sine(250.0)] * 3

        gate = check_voice(segments, OUTPUT_SAMPLE_RATE, baseline_p10=P10_OF_200HZ)

        assert gate.drift_pct == pytest.approx(
            100.0 * (P10_OF_250HZ - P10_OF_200HZ) / P10_OF_200HZ
        )
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
