"""synth 里不需要模型就能测的部分：参考音/参考文本的成对校验，以及三个「库变了」的守卫。

这条校验挡的是设计 §4 里最贵的错误——只传音频不传文本，模型会静默切换到
continuation 模式，声音换人但一切「正常」。

守卫那几条测试用假模型跑，因为它们要测的恰恰是**真实模型永远不会出现的形状**
（缺 patch_size、一次 yield 都没有）。真实模型只能证明守卫没被误触发，证不了
它触发时的行为对不对。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from voxout import wavio
from voxout.synth import Synthesizer, resolve_reference


@dataclass(frozen=True)
class FakeResult:
    """库的 GenerationResult 里 synth 真正读的那几个字段。"""

    audio: np.ndarray
    processing_time_seconds: float = 0.5
    token_count: int = 4
    peak_memory_usage: float = 3.0


class FakeVae:
    decode_chunk_size = 1920


class FakeModel:
    """按 synth 实际用到的接口造的替身，缺什么由参数决定。"""

    sample_rate = 48000
    patch_size = 4

    def __init__(self, *results: FakeResult) -> None:
        self.audio_vae = FakeVae()
        self._results = results
        self.calls: list[dict[str, object]] = []

    def generate(self, text: str, **kwargs: object) -> tuple[FakeResult, ...]:
        self.calls.append({"text": text, **kwargs})
        return self._results


class ModelWithoutPatchSize(FakeModel):
    """模拟库改名/改结构。取错这两个数会导致静默截断，所以必须抛而不是回退常量。"""

    def __init__(self) -> None:
        super().__init__()
        del self.audio_vae


def _synthesizer_holding(model: object) -> Synthesizer:
    """装一个假模型，绕过 __init__。

    `__init__` 唯一的工作是加载 3.0GB 权重。绕开它的代价是这个辅助函数知道了
    `_model` / `_model_id` 两个私有名字；换来的是三个守卫可以在毫秒级被验证。
    为此在生产代码上开一个注入口才是更坏的交易（YAGNI）。
    """
    synthesizer = object.__new__(Synthesizer)
    synthesizer._model = model
    synthesizer._model_id = "fake/VoxCPM2"
    return synthesizer


@pytest.fixture
def anchor(tmp_path: Path) -> Path:
    path = tmp_path / "anchor.wav"
    wavio.write(path, np.zeros(4800, dtype=np.float32), 48000)
    return path


def test_both_absent_means_no_cloning(anchor: Path) -> None:
    assert resolve_reference(None, None) == (None, None)


def test_both_present_returns_a_path_string_and_trimmed_text(anchor: Path) -> None:
    """库接受路径字符串并自行重采样到 16kHz，所以这里给 str 而不是 Path。"""
    assert resolve_reference(anchor, "  锚点文本。 ") == (str(anchor), "锚点文本。")


def test_audio_without_text_is_refused(anchor: Path) -> None:
    with pytest.raises(ValueError, match="成对"):
        resolve_reference(anchor, None)


def test_text_without_audio_is_refused() -> None:
    with pytest.raises(ValueError, match="成对"):
        resolve_reference(None, "锚点文本。")


def test_blank_text_counts_as_missing(anchor: Path) -> None:
    with pytest.raises(ValueError, match="成对"):
        resolve_reference(anchor, "   \n")


def test_missing_audio_file_is_reported_with_its_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.wav"
    with pytest.raises(FileNotFoundError, match="nope.wav"):
        resolve_reference(missing, "锚点文本。")


class TestGuardsAgainstTheLibraryChanging:
    """三条守卫，都对应「静默产出错东西」而不是「报错」。"""

    def test_patch_geometry_is_read_from_the_model(self) -> None:
        assert _synthesizer_holding(FakeModel()).seconds_per_patch == pytest.approx(0.16)

    def test_missing_patch_geometry_raises_instead_of_falling_back(self) -> None:
        """回退到常量 0.16 会在库改了几何之后算出错的预算 → 静默截断，
        而静默截断正是选型文档里 Kokoro 那个丢 27% 内容不报错的失败模式。"""
        with pytest.raises(RuntimeError, match="token 预算"):
            _ = _synthesizer_holding(ModelWithoutPatchSize()).seconds_per_patch

    def test_a_model_that_yields_nothing_is_an_error_not_empty_audio(self) -> None:
        """空音频往下传会变成一段静音，拼接、写盘、闸门 1 都不会崩——
        直到有人听成品才发现少了一段。"""
        with pytest.raises(RuntimeError, match="没有产出任何音频"):
            _synthesizer_holding(FakeModel()).generate("一段文本。")

    def test_multiple_yields_are_concatenated_in_order(self) -> None:
        """`generate` 的注释声明「不假设 yield 次数」，这里把那句话钉住。

        当前库单次 yield，所以真实模型测不到这条；库哪天改成分块 yield，
        少拼一块就是少一段音频。
        """
        head = np.full(4800, 0.1, dtype=np.float32)
        tail = np.full(2400, -0.2, dtype=np.float32)
        model = FakeModel(FakeResult(head), FakeResult(tail))

        segment = _synthesizer_holding(model).generate("一段文本。")

        assert len(segment.audio) == len(head) + len(tail)
        assert segment.audio[0] == pytest.approx(0.1)
        assert segment.audio[-1] == pytest.approx(-0.2)
        assert segment.audio_seconds == pytest.approx(7200 / 48000)
        assert segment.token_count == 8, "各次 yield 的 token 数要相加，不是取最后一次"
        assert segment.generation_seconds == pytest.approx(1.0)

    def test_model_id_is_exposed_for_the_manifest(self) -> None:
        """manifest 要记模型来源，否则一批音频事后无法归因到权重版本。"""
        assert _synthesizer_holding(FakeModel()).model_id == "fake/VoxCPM2"
