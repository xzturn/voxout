"""单测共用的替身与实测常量。不被 pytest 收集（文件名不匹配 test_*）。

这些替身存在的理由是：两道闸门、渲染流水线与 CLI 的逻辑都必须能在**不加载
任何模型**的前提下测完。加载一次 VoxCPM2 要 1.2–2.0 秒、3.0GB 权重，而这些
逻辑与模型无关——把它们绑到模型上会让快速回路变成分钟级，人就不再频繁跑它。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxout.config import OUTPUT_SAMPLE_RATE
from voxout.synth import Segment

# 自相关峰值落在整数延迟格上，Hann 窗又让自相关随延迟衰减，峰值被拉向短一格：
# 200 Hz 的理论延迟 240 → 实取 239 → 48000/239 = 200.84。偏差上界约 1/延迟
# （本频段 0.4–0.6%），是系统性的，而真实基线由同一函数在真实音频上测得，
# 所以在闸门里同偏同消。用实测值而非标称值，才能把「零漂移」测成真正的零。
P10_OF_200HZ = 48000 / 239
P10_OF_250HZ = 48000 / 191

FAKE_SEGMENT_SECONDS = 0.6
"""替身每次调用产出的固定时长。

**刻意不与文本长度成正比**：切段测试要送上千汉字，按 0.3 秒/汉字算出来的
音频会让 p10_f0 内部那次 rfft 分配数百 MB。而「时长与汉字数成正比」这件事
本来也不该由替身来验——真实速率是 T13 在真实模型上测的（0.23–0.25 秒/汉字）。
"""


def sine(
    freq: float = 200.0, seconds: float = 2.0, sr: int = OUTPUT_SAMPLE_RATE
) -> np.ndarray:
    """一段纯音。p10 F0 可精确预测，是验证闸门 2 接线的理想信号。"""
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


class FakeSynthesizer:
    """替身：产出可预测音高的纯音，并记录每次调用的参考音与参考文本。

    `freq` 可变是为了模拟音色漂移：渲染流水线必须能在「模型换了个人说话」时
    让闸门 2 失败，而这在真实模型上无法按需触发。
    """

    def __init__(
        self,
        *,
        freq: float = 200.0,
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        model_id: str = "fake/VoxCPM2",
    ) -> None:
        self.freq = freq
        self._sample_rate = sample_rate
        self._model_id = model_id
        self.calls: list[dict[str, object]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def seconds_per_patch(self) -> float:
        return 0.16

    def generate(
        self,
        text: str,
        *,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> Segment:
        self.calls.append(
            {
                "text": text,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "seed": seed,
                "max_tokens": max_tokens,
            }
        )
        seconds = FAKE_SEGMENT_SECONDS
        audio = sine(self.freq, seconds, self._sample_rate)
        return Segment(
            audio=audio,
            sample_rate=self._sample_rate,
            audio_seconds=len(audio) / self._sample_rate,
            generation_seconds=0.5 * seconds,
            token_count=int(seconds / 0.16),
            max_tokens=max_tokens or 4000,
            peak_memory_gb=3.0,
        )
