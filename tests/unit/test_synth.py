"""synth 里不需要模型就能测的部分：参考音/参考文本的成对校验。

这条校验挡的是设计 §4 里最贵的错误——只传音频不传文本，模型会静默切换到
continuation 模式，声音换人但一切「正常」。
"""

from pathlib import Path

import numpy as np
import pytest

from voxout import wavio
from voxout.synth import resolve_reference


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
