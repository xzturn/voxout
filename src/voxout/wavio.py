"""单声道 16-bit PCM wav 的读写与探查。

用标准库 `wave` 而不是 soundfile：我们自己只产出一种格式（48kHz 单声道 int16），
读的也只有自己写的锚点与渲染产物。soundfile 已被 mlx-audio 拉进来用于读模型
的输入，但让我们自己的 I/O 依赖它没有收益——标准库够了，且行为完全可预测。
"""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_INT16_PEAK = 32767
_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class WavInfo:
    """wav 的几何信息。用于校验，不加载样本。"""

    sample_rate: int
    channels: int
    sample_width: int
    n_frames: int

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.sample_rate


def probe(path: Path) -> WavInfo:
    """只读文件头。校验一个 3 分钟的锚点不必把它整个读进内存。"""
    with wave.open(str(path), "rb") as handle:
        return WavInfo(
            sample_rate=handle.getframerate(),
            channels=handle.getnchannels(),
            sample_width=handle.getsampwidth(),
            n_frames=handle.getnframes(),
        )


def read(path: Path) -> np.ndarray:
    """读成 float32 单声道，范围 [-1, 1]。

    Raises:
        ValueError: 非单声道或非 16-bit。静默取左声道或猜测位宽会悄悄改变
            音色度量的结果，那比明确失败危险得多。
    """
    info = probe(path)
    if info.channels != 1:
        raise ValueError(f"{path} 是 {info.channels} 声道，只接受单声道")
    if info.sample_width != _SAMPLE_WIDTH_BYTES:
        raise ValueError(f"{path} 是 {info.sample_width * 8}-bit，只接受 16-bit PCM")

    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(info.n_frames)

    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_PEAK).copy()


def write(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """写成 48kHz 单声道 int16 wav。

    越界样本先裁剪再转换：float32 直接转 int16 时超过 ±1.0 的值会整数回绕，
    符号翻转变成响亮的爆音。模型输出偶尔会略微越界，所以这不是理论问题。

    Raises:
        ValueError: 音频为空、维度不对，或含 nan/inf。
    """
    if audio.ndim != 1:
        raise ValueError(f"只接受一维单声道音频，收到 shape={audio.shape}")
    if audio.size == 0:
        raise ValueError("音频为空，不写出静默的空文件")
    if not np.isfinite(audio).all():
        raise ValueError("音频含 nan 或 inf，必须全部为有限值")

    samples = (np.clip(audio, -1.0, 1.0) * _INT16_PEAK).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(_SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
