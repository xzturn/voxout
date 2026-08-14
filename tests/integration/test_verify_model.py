"""闸门在真实 TTS + ASR 回环上的表现。慢（两个模型），故标记 integration。

单测用假 transcriber 验的是判定逻辑；这里验的是**回环本身闭合**——真实模型
读出来的音，真实 ASR 能听回同样的内容。这两件事都可能单独正确而合起来失败。
"""

from pathlib import Path

import numpy as np
import pytest

from voxout import wavio
from voxout.config import COVERAGE_MIN
from voxout.metrics import p10_f0
from voxout.normalize import to_speakable_numbers
from voxout.synth import Synthesizer
from voxout.textmatch import coverage
from voxout.verify import check_content, check_voice, load_transcriber

pytestmark = pytest.mark.integration

NUMERIC_SOURCE = "2026 年 8 月，注意力下降了 12%，用时 1,200 毫秒。"


@pytest.fixture(scope="module")
def synthesizer() -> Synthesizer:
    return Synthesizer()


@pytest.fixture(scope="module")
def transcriber() -> object:
    return load_transcriber()


@pytest.fixture(scope="module")
def numeric_render(synthesizer: Synthesizer, tmp_path_factory: pytest.TempPathFactory):
    """一段含年份、百分比、千分位与单位的音频——归一化四条前置规则全覆盖。"""
    spoken = to_speakable_numbers(NUMERIC_SOURCE)
    segment = synthesizer.generate(spoken, seed=42)
    path = tmp_path_factory.mktemp("verify") / "numeric.wav"
    wavio.write(path, segment.audio, segment.sample_rate)
    return spoken, path, segment


def test_faithful_render_passes_the_content_gate(numeric_render, transcriber) -> None:
    """实测 coverage = 1.0000。

    这不是宽松通过，是精确闭合：朗读侧是
    「二零二六年八月，注意力下降了百分之十二，用时一千二百毫秒。」
    ASR 侧回的是「2026年8月,注意力下降了12%,用时1200毫秒。」——字面几乎不重叠，
    双侧归一化后完全一致。
    """
    spoken, path, _ = numeric_render

    gate = check_content(path, spoken, transcriber)

    assert gate.passed
    assert gate.coverage >= COVERAGE_MIN
    assert not gate.asr_normalize_failed


def test_double_sided_normalization_is_what_makes_it_pass(
    numeric_render, transcriber
) -> None:
    """第四条修正在真实 ASR 输出上的锁。

    实测单侧比较得 0.4615——一次完美的渲染会被判成丢了 54% 的内容，
    而人会去修一个没坏的东西。这条测试用真实转写结果证明差距存在。
    """
    spoken, path, _ = numeric_render
    gate = check_content(path, spoken, transcriber)

    naive = coverage(spoken, gate.asr_text)

    assert naive < COVERAGE_MIN, "若裸比较也能过闸，这条修正就没有必要了"
    assert gate.coverage > naive


def test_real_speech_yields_a_finite_pitch_floor(numeric_render) -> None:
    """真实合成语音必须测得出浊音基频——否则闸门 2 永远在比 nan。

    合成语音不保证有可检测的浊音帧（清音过多、能量过低都会让它落空），
    所以这条不是重言式。
    """
    _, _, segment = numeric_render

    baseline = p10_f0(segment.audio, segment.sample_rate)

    assert np.isfinite(baseline)
    assert 60.0 < baseline < 400.0


def test_voice_gate_closes_on_real_audio(numeric_render) -> None:
    """用同一段音频自己测出的基线判它自己：漂移应为精确的 0。

    验的是 check_voice 与 p10_f0 之间的接线（采样率有没有传对、段有没有被
    错切），而不是模型的音色稳定性——后者要靠锚点，是 T13 的事。
    """
    _, _, segment = numeric_render
    baseline = p10_f0(segment.audio, segment.sample_rate)

    gate = check_voice([segment.audio], segment.sample_rate, baseline_p10=baseline)

    assert gate.passed
    assert gate.drift_pct == pytest.approx(0.0, abs=1e-9)
    assert gate.spread_pct == 0.0


def test_missing_wav_is_refused_before_loading_audio(
    tmp_path: Path, transcriber
) -> None:
    with pytest.raises(FileNotFoundError):
        check_content(tmp_path / "absent.wav", "文本", transcriber)
