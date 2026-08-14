"""碰真实模型的集成测试。慢（每次加载 1.2–2.0 秒 + 生成），故标记 integration。

快速回路：uv run pytest -m "not integration"
"""

import numpy as np
import pytest

from voxout.config import OUTPUT_SAMPLE_RATE, SECONDS_PER_PATCH_NOMINAL
from voxout.synth import Synthesizer

pytestmark = pytest.mark.integration

SHORT_TEXT = "这是一段测试。"


@pytest.fixture(scope="module")
def synthesizer() -> Synthesizer:
    """整个模块共用一个实例——加载是最贵的一步，不该按测试重复付。"""
    return Synthesizer()


def test_output_is_native_forty_eight_kilohertz(synthesizer: Synthesizer) -> None:
    assert synthesizer.sample_rate == OUTPUT_SAMPLE_RATE


def test_patch_duration_matches_the_documented_nominal(synthesizer: Synthesizer) -> None:
    """patch_size(4) × decode_chunk_size(1920) / 48000 = 0.16 秒。

    这个数是 token 预算的基础。库若改了 patch 几何，预算会整体偏移，
    表现为长文静默截断——所以要有一条测试直接盯住它。
    """
    assert synthesizer.seconds_per_patch == pytest.approx(SECONDS_PER_PATCH_NOMINAL)


def test_short_text_produces_plausible_audio(synthesizer: Synthesizer) -> None:
    segment = synthesizer.generate(SHORT_TEXT, seed=42)

    assert segment.sample_rate == OUTPUT_SAMPLE_RATE
    assert segment.audio_seconds > 0.5
    assert segment.token_count > 0
    assert segment.generation_seconds > 0
    assert segment.audio.dtype == np.float32
    assert np.isfinite(segment.audio).all()


def test_budget_is_derived_and_exceeds_what_the_text_needs(
    synthesizer: Synthesizer,
) -> None:
    segment = synthesizer.generate(SHORT_TEXT, seed=42)

    # 预算按 0.30 秒/汉字的保守上界算，实际约 0.21，所以必须留有余量。
    assert segment.max_tokens * segment.sample_rate * SECONDS_PER_PATCH_NOMINAL > 0
    assert segment.audio_seconds < segment.max_tokens * synthesizer.seconds_per_patch


def test_same_seed_and_text_reproduce_bit_identical_audio(
    synthesizer: Synthesizer,
) -> None:
    """设计 §3 实测的性质，锚点的可复现性依赖它。

    若这条失败，`mint-voice` 就不再是幂等的，档案里的基线也不再代表锚点音频。
    """
    first = synthesizer.generate(SHORT_TEXT, seed=42)
    second = synthesizer.generate(SHORT_TEXT, seed=42)

    assert len(first.audio) == len(second.audio)
    assert np.abs(first.audio - second.audio).max() == 0.0


def test_empty_text_is_refused_before_touching_the_model(
    synthesizer: Synthesizer,
) -> None:
    with pytest.raises(ValueError):
        synthesizer.generate("   ")
