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
    """各段对基线漂移**绝对值的中位数**。判定就看这个，理由见 `check_voice`。

    取绝对值再求中位数，而不是先求带符号中位数再取绝对值：后者在各段一半 +30%
    一半 −30% 时读出约 0，把最不稳的情形判成最稳。
    """
    spread_pct: float
    """段间极差，旁证。单段时为 0；有效值不足两个时为 nan。

    **不要把它升级成判据。** 极差和最大值一样是极值统计量：既随段数上升（6 段
    12.38%、14 段 27.50%），也在同一篇帖子的两次生成之间大幅跳动（213 汉字那篇
    12.38% → 40.39%，同期中位漂移只从 5.35% 动到 5.02%）。拿它当闸门会重新引入
    `check_voice` 里刚去掉的那个缺陷，而且还会随机误报。

    它的用处是在闸门**已经**判失败时分辨形状：漂移大而极差小 = 整体换了人；
    漂移大而极差也大 = 某几段跑了，去听那几段。正常范围本来就宽（实测 12–40%），
    所以它单独出现时不构成任何信号。
    """
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
    是它的推论。

    **判据取各段漂移的中位数，不取最大值。** 原先取最大值，等于让全篇的判定由
    单独一段决定，而那个数是随段数长的：同一个音色、同一个基线，6 段的帖子读出
    max 10.65%，14 段读出 13.79%，中位数则是 5.35% 与 4.72%（T13 实测，两篇真实
    短帖）。极值统计量的期望本来就随抽样数上升，于是「文章更长」会被读成「音色
    更飘」——判据依赖输入长度，这是定义缺陷，不是阈值不够宽。

    换成中位数之后稳定性可以直接看出来：两篇帖子各生成两次，四个读数是 5.35%、
    5.02%、4.72%、5.04%，总极差 0.6 个百分点，且与段数（6 与 14）无关。同期
    `spread_pct` 在同一篇帖子的两次生成之间从 12.38% 跳到 40.39%。

    单段偏离十几个百分点在真实内容里是常态（cal2 第 11 段 +12.6%、第 13 段
    −13.8%，音色由人听一致），因为 p10 是 F0 分布的低端分位，随句子长短与语调
    走向变化。中位数只在**多数**段偏离时才动，而「换了个人」恰好是多数段一起偏
    （零样本实测 48.82%–53.9%）。代价是单段级的音色事故不再由闸门拦住，只以
    `segment_p10s` 与 `spread_pct` 呈现给人——闸门 2 的失败本就是建议而非中止
    （模块文档约束 1），单段识别属于 LTAS 那条待办。

    用 `np.median` 而不是 `statistics.median`：numpy 对含 nan 的输入确定地返回
    nan，而 `nan <= 阈值` 恒为 False，静音段于是照旧失败。标准库版本会把 nan
    参与排序，结果取决于输入顺序，静音段能否被发现会变成参数顺序的偶然。

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
    drift = float(np.median(np.abs(np.asarray(p10s) - baseline_p10) / baseline_p10 * 100.0))

    return VoiceGate(
        baseline_p10=baseline_p10,
        segment_p10s=p10s,
        drift_pct=drift,
        spread_pct=_inter_segment_spread(p10s),
        threshold=VOICE_DRIFT_MAX_PCT,
        passed=drift <= VOICE_DRIFT_MAX_PCT,
    )
