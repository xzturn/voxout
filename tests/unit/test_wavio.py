"""wav 读写。用标准库 wave，不引入 soundfile——我们只处理单声道 16-bit PCM。"""

import wave
from pathlib import Path

import numpy as np
import pytest

from voxout import wavio


def test_round_trip_preserves_length_and_rate(tmp_path: Path) -> None:
    audio = np.sin(np.arange(4800) * 0.1).astype(np.float32)
    path = tmp_path / "a.wav"

    wavio.write(path, audio, 48000)
    restored = wavio.read(path)

    assert len(restored) == len(audio)
    assert wavio.probe(path).sample_rate == 48000
    assert restored.dtype == np.float32


def test_round_trip_is_accurate_to_int16_quantization(tmp_path: Path) -> None:
    audio = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    path = tmp_path / "a.wav"

    wavio.write(path, audio, 48000)

    # int16 的量化步长是 1/32767 ≈ 3.05e-5，误差不应超过半个步长。
    assert np.abs(wavio.read(path) - audio).max() < 1.0 / 32767


def test_out_of_range_samples_are_clipped_not_wrapped(tmp_path: Path) -> None:
    """超过 ±1.0 的样本若直接转 int16 会整数回绕，变成响亮的爆音。

    模型输出偶尔会略微越界，所以这条不是理论问题。回绕后的符号会翻转，
    这里断言符号保持为正。
    """
    path = tmp_path / "a.wav"
    wavio.write(path, np.array([1.5, -1.5], dtype=np.float32), 48000)

    restored = wavio.read(path)
    assert restored[0] > 0.99
    assert restored[1] < -0.99


def test_probe_reports_geometry_without_loading(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    wavio.write(path, np.zeros(48000, dtype=np.float32), 48000)

    info = wavio.probe(path)
    assert (info.sample_rate, info.channels, info.sample_width) == (48000, 1, 2)
    assert info.duration_seconds == pytest.approx(1.0)


def test_reading_stereo_is_refused(tmp_path: Path) -> None:
    """立体声要在这里明确失败——静默取左声道会悄悄改变音色度量。"""
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(np.zeros(2000, dtype=np.int16).tobytes())

    with pytest.raises(ValueError, match="单声道"):
        wavio.read(path)


def test_writing_non_finite_samples_is_refused(tmp_path: Path) -> None:
    """nan/inf 转 int16 的结果由平台决定，静默写出会得到无法解释的音频。"""
    with pytest.raises(ValueError, match="有限"):
        wavio.write(tmp_path / "a.wav", np.array([0.1, np.nan], dtype=np.float32), 48000)


def test_writing_empty_audio_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        wavio.write(tmp_path / "a.wav", np.array([], dtype=np.float32), 48000)
