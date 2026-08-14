"""唯一碰生成模型的模块。

一次渲染只加载一个模型实例（实测每进程加载 1.22–1.95 秒）。多段长文共用它，
这也保证了段与段之间用的是同一份权重。

三个刻意不信库的地方：

1. `GenerationResult.real_time_factor` 是 `音频时长/耗时`，与选型文档定义的
   RTF（耗时/音频时长）互为倒数。只记原始的 `generation_seconds` 与
   `audio_seconds`，让消费方自己派生，避免语义反转。
2. `GenerationResult.audio_duration` 是 `"HH:MM:SS.mmm"` 字符串，不是数字。
   时长由 `samples / sample_rate` 自己算。
3. `patch_size` / `decode_chunk_size` 从模型实例读，不用常量。这两个数决定
   token 预算，取错会直接导致静默截断。
"""

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from voxout.budget import plan_max_tokens, seconds_per_patch
from voxout.config import MODEL_ID
from voxout.hub import load_or_explain
from voxout.textmatch import han_only


@dataclass(frozen=True)
class Segment:
    """一次 generate 调用的产物与它的全部计量。"""

    audio: np.ndarray
    sample_rate: int
    audio_seconds: float
    generation_seconds: float
    token_count: int
    max_tokens: int
    peak_memory_gb: float


def resolve_reference(
    ref_audio: Path | None, ref_text: str | None
) -> tuple[str, str] | tuple[None, None]:
    """校验参考音与参考文本必须成对出现。

    只给音频不给文本，模型会走「continuation」模式而不是克隆模式；只给文本
    不给音频则完全不克隆。两种都会静默换掉声音——设计 §4 实测这两条路的
    p10 F0 漂移分别是 5.2% 和 14.7%，而无锚点是 53.9%。

    Raises:
        ValueError: 只给了其中一个。
        FileNotFoundError: 音频文件不存在。
    """
    if ref_audio is None and ref_text is None:
        return None, None

    if ref_audio is None or ref_text is None or not ref_text.strip():
        raise ValueError(
            "ref_audio 与 ref_text 必须成对提供。缺一个会让模型静默切换到"
            "非克隆模式（实测 p10 F0 漂移 5.2%–53.9%），声音就换人了。"
        )

    if not ref_audio.is_file():
        raise FileNotFoundError(f"参考音频不存在：{ref_audio}")

    return str(ref_audio), ref_text.strip()


class Synthesizer:
    """持一个已加载的 VoxCPM2 实例。"""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        from mlx_audio.tts.utils import load

        self._model = load_or_explain(load, model_id, "mlx_audio.tts.utils")
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    @property
    def seconds_per_patch(self) -> float:
        """每个 patch 对应的音频时长，从模型配置派生。

        缺字段就抛错而不是回退到常量：预算算错的后果是静默截断，
        而静默截断正是选型文档里 Kokoro 那个失败模式（丢 27% 内容不报错）。
        """
        try:
            patch_size = int(self._model.patch_size)
            decode_chunk_size = int(self._model.audio_vae.decode_chunk_size)
        except AttributeError as exc:
            raise RuntimeError(
                f"模型 {self._model_id} 没有 patch_size / audio_vae.decode_chunk_size，"
                f"无法推算 token 预算。库版本可能变了，需要重新核对：{exc}"
            ) from exc

        return seconds_per_patch(patch_size, decode_chunk_size, self.sample_rate)

    def generate(
        self,
        text: str,
        *,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
    ) -> Segment:
        """把一段文本合成为音频。

        `max_tokens` 缺省时由汉字数推算——显式给出预算，把截断从「事后由闸门 1
        发现」变成「事前由 BudgetError 排除」。

        Raises:
            ValueError: 文本为空，或参考音与参考文本未成对。
            BudgetError: 文本超出单次调用的 token 上限（应先切段）。
        """
        if not text.strip():
            raise ValueError("待合成文本为空")

        reference_path, reference_text = resolve_reference(ref_audio, ref_text)
        budget = max_tokens or plan_max_tokens(
            max(len(han_only(text)), 1), self.seconds_per_patch
        )

        if seed is not None:
            mx.random.seed(seed)

        results = list(
            self._model.generate(
                text,
                max_tokens=budget,
                ref_audio=reference_path,
                ref_text=reference_text,
            )
        )
        if not results:
            raise RuntimeError(f"模型对 {len(text)} 字的输入没有产出任何音频")

        # 当前实现单次 yield，但拼接对多次 yield 同样正确——不假设 yield 次数。
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
        return Segment(
            audio=audio,
            sample_rate=self.sample_rate,
            audio_seconds=len(audio) / self.sample_rate,
            generation_seconds=sum(float(r.processing_time_seconds) for r in results),
            token_count=sum(int(r.token_count) for r in results),
            max_tokens=budget,
            peak_memory_gb=max(float(r.peak_memory_usage) for r in results),
        )
