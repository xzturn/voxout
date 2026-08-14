"""两道闸门：内容完整性与音色一致性。

这两道闸门存在是因为 TTS 的两种主要失效都是**静默**的——不抛异常、不打日志，
只是成品悄悄变坏，而人往往在发布之后才发现：

  闸门 1（ASR 回环）  挡静默丢字。选型阶段 Kokoro 丢了 27% 的内容而全程「正常」。
  闸门 2（p10 F0）    挡静默换声。零样本模式下同一段文本两次调用的 p10 F0
                      极差 53.9%，等于每次换一个人说话（设计 §4）。

两条设计约束：

1. **失败是返回值，不是异常。** 音频仍要落盘。判断「模型读漏了」还是
   「ASR 听错了」只能靠人听一遍，把失败变成异常会连证据一起丢掉。
2. **判定必须可复现。** 闸门若时好时坏就不再是闸门。所以 ASR 的语言与温度
   都显式钉住，而不是用库的默认值。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from voxout import normalize
from voxout.config import ASR_MODEL_ID, COVERAGE_MIN, VOICE_DRIFT_MAX_PCT
from voxout.hub import load_or_explain
from voxout.metrics import p10_f0, spread_pct
from voxout.textmatch import coverage, missing_spans


@dataclass(frozen=True)
class ContentGate:
    """闸门 1 的判定与它的全部依据。"""

    coverage: float
    threshold: float
    passed: bool
    asr_text: str
    """ASR 的**原话**，不是归一化后的版本——排查时人看的是原话。"""
    missing: tuple[tuple[int, str], ...]
    """(汉字序号, 缺失片段)。只说「不通过」没用，得指出丢在哪一段。"""
    asr_normalize_failed: bool
    """归一化本身是否出过错。区分「内容真的丢了」与「比较工具坏了」。"""


@dataclass(frozen=True)
class VoiceGate:
    """闸门 2 的判定与它的全部依据。"""

    baseline_p10: float
    segment_p10s: tuple[float, ...]
    drift_pct: float
    """各段对基线漂移的最大值。判定就看这个。"""
    spread_pct: float
    """段间极差，旁证。单段时为 0；有效值不足两个时为 nan。"""
    threshold: float
    passed: bool


class Transcription(Protocol):
    text: str


class TranscriberLike(Protocol):
    """check_content 对 ASR 的全部要求。单测传替身即可，不必加载模型。"""

    def generate(
        self, audio: str, *, language: str = ..., temperature: float = ...
    ) -> Transcription: ...


def load_transcriber(model_id: str = ASR_MODEL_ID) -> TranscriberLike:
    """加载 ASR 模型。只有闸门 1 用它，渲染本身不需要。"""
    from mlx_audio.stt.utils import load

    return load_or_explain(load, model_id, "mlx_audio.stt.utils")


def check_content(
    wav_path: Path, spoken_text: str, transcriber: TranscriberLike
) -> ContentGate:
    """把音频转写回文本，比对汉字覆盖率。

    **两侧都过归一化再比**：朗读文本已是「第一百二十二天」，而 ASR 按听到的音
    回写成「第122天」。内容完全一致，字面却只共享「第」「天」两字——单侧比较
    实测得 0.2857，会把一次完美的渲染判成丢了 71% 的内容。

    Raises:
        FileNotFoundError: 音频不存在。
        ValueError: 待校验文本为空。
    """
    if not wav_path.is_file():
        raise FileNotFoundError(f"待校验音频不存在：{wav_path}")
    if not spoken_text.strip():
        raise ValueError("待校验文本为空，覆盖率无从计算")

    # 显式钉住语言与温度：温度默认是回退元组 (0.0, 0.2, …, 1.0)，一旦触发回退，
    # 同一段音频两次转写可能给出不同数字，闸门就不再可复现。
    heard = transcriber.generate(str(wav_path), language="zh", temperature=0.0).text

    heard_normalized, normalized_ok = normalize.to_speakable_numbers_checked(heard)
    # 朗读侧在渲染时已归一化过一次（幂等），这里不再单独报告它的成败。
    reference = normalize.to_speakable_numbers(spoken_text)

    ratio = coverage(reference, heard_normalized)
    return ContentGate(
        coverage=ratio,
        threshold=COVERAGE_MIN,
        passed=ratio >= COVERAGE_MIN,
        asr_text=heard,
        missing=missing_spans(reference, heard_normalized),
        asr_normalize_failed=not normalized_ok,
    )


def _inter_segment_spread(p10s: tuple[float, ...]) -> float:
    """段间极差。单段没有段间之说（定义为 0）；有效值不足两个时无从比较（nan）。"""
    if len(p10s) < 2:
        return 0.0
    if sum(1 for value in p10s if np.isfinite(value)) < 2:
        return float("nan")
    return spread_pct(list(p10s))


def check_voice(
    segments: Sequence[np.ndarray], sr: int, baseline_p10: float
) -> VoiceGate:
    """比对每段的浊音音高地板与档案基线的偏离。

    判据是对基线的漂移而非段间极差：基线代表我们承诺复现的那个人，而段间一致
    是它的推论——各段都在基线 ±6% 内，彼此之间就不会超过约 12%。

    用 `np.max` 而不是内置 `max`：内置 `max` 遇 nan 的结果取决于比较顺序
    （`max(nan, 5)` 给 nan 而 `max(5, nan)` 给 5），静音段能否被发现会变成
    参数顺序的偶然。`np.max` 确定地传播 nan，而 `nan <= 阈值` 恒为 False。

    Raises:
        ValueError: 没有音频段，或基线非正/非有限（它是漂移百分比的分母）。
    """
    if len(segments) == 0:
        raise ValueError("没有音频段可校验")
    if not np.isfinite(baseline_p10) or baseline_p10 <= 0:
        raise ValueError(
            f"基线 p10 F0 非法（{baseline_p10!r}）。它是漂移百分比的分母，"
            f"为 0 或 nan 会让判定失去意义。"
        )

    p10s = tuple(p10_f0(np.asarray(s, dtype=np.float32), sr) for s in segments)
    drift = float(np.max(np.abs(np.asarray(p10s) - baseline_p10) / baseline_p10 * 100.0))

    return VoiceGate(
        baseline_p10=baseline_p10,
        segment_p10s=p10s,
        drift_pct=drift,
        spread_pct=_inter_segment_spread(p10s),
        threshold=VOICE_DRIFT_MAX_PCT,
        passed=drift <= VOICE_DRIFT_MAX_PCT,
    )
