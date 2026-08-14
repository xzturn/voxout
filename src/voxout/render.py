"""渲染流水线：markdown 进，带证据的音频出。

编排顺序由一个约束决定：**闸门 1 要读文件才能转写**。所以音频先落到临时文件，
过了闸门才改名成成品。这样任何一次失败都不会在成品目录里留下半个文件——
一个看起来正常、听起来才发现丢了内容的 wav 是最坏的产物。

两道闸门的处置刻意不对称（设计 §9）：

  闸门 1 内容  不过 → 不出音频，写差异报告。内容丢失是客观的。
  闸门 2 音色  不过 → 仍出音频，manifest 标记。判据是代理指标（p10 F0），
               可能误报，必须让人能听到那段音频才判得了。

失败以返回值表达而非异常，理由同 verify：证据要落盘，人要能看。真正的异常
只留给「无法继续」——文件不存在、清洗后无内容、采样率不符。
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voxout import assemble, chunker, markdown, normalize, wavio
from voxout.config import ASR_MODEL_ID, OUTPUT_DIR, OUTPUT_SAMPLE_RATE
from voxout.profile import VoiceProfile
from voxout.synth import Segment
from voxout.verify import ContentGate, TranscriberLike, VoiceGate, check_content, check_voice

_LOG = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_PARTIAL_SUFFIX = ".partial.wav"
_MANIFEST_SUFFIX = ".manifest.json"
_DIFF_SUFFIX = ".diff.txt"


class SynthLike(Protocol):
    """渲染对合成器的全部要求。单测传替身即可，不必加载 3.0GB 权重。"""

    @property
    def model_id(self) -> str: ...

    @property
    def sample_rate(self) -> int: ...

    def generate(
        self,
        text: str,
        *,
        ref_audio: Path | None = ...,
        ref_text: str | None = ...,
    ) -> Segment: ...


@dataclass(frozen=True)
class RenderResult:
    """一次渲染的结果与它落盘的全部文件。

    `audio_path is None` 表示音频被有意丢弃（闸门 1 不过且未 --force）；
    `voice_gate is None` 表示它没被测——内容错了，音色读数没有意义。
    """

    source: Path
    voice: str
    spoken_text: str
    audio_path: Path | None
    manifest_path: Path | None
    diff_path: Path | None
    content_gate: ContentGate
    voice_gate: VoiceGate | None
    manifest: dict[str, object] | None


def default_output(source: Path, directory: Path = OUTPUT_DIR) -> Path:
    """`out/<帖子名>.wav`。中文保留，空白折成连字符——带空格的文件名每次引用都要加引号。"""
    return directory / f"{_WHITESPACE.sub('-', source.stem.strip())}.wav"


def _prepare_text(source: Path) -> str:
    """md → 朗读文本。归一化在**送进模型之前**做，不只在闸门比较时做。

    Raises:
        FileNotFoundError: 源文件不存在。
        ValueError: 清洗后无可朗读内容。
    """
    if not source.is_file():
        raise FileNotFoundError(f"源文件不存在：{source}")

    try:
        speakable = markdown.to_speakable(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        # 补上只有这一层知道的两件事：是哪个文件，以及为什么宁可失败。
        # 批量渲染时不带文件名的报错等于没报。
        raise ValueError(
            f"{source}：{exc}。产出一个静音文件比明确失败糟得多——"
            f"它有正确的文件名和正确的时长，听了才发现是空的。"
        ) from exc

    return normalize.to_speakable_numbers(speakable)


def _synthesize(
    spoken_text: str, profile: VoiceProfile, synthesizer: SynthLike
) -> tuple[Segment, ...]:
    """逐段生成。每段都带上锚点——漏一段，那一段就换个人说话（设计 §4：53.9%）。"""
    if synthesizer.sample_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(
            f"合成器输出 {synthesizer.sample_rate} Hz，要求 {OUTPUT_SAMPLE_RATE} Hz。"
            f"按错的采样率解读音频会让 F0 系统性偏移，表现为「音色漂移」"
            f"而真实原因是单位错。"
        )

    chunks = chunker.chunk(spoken_text)
    _LOG.info("切成 %d 段，共 %d 字", len(chunks), len(spoken_text))

    # 逐段记进度，而不是一次性 tuple(...)：长文要生成好几分钟，没有进度输出的
    # 等待与「挂死」在终端上是同一种样子，而挂死正是设计 §9 要防的那类失败。
    segments: list[Segment] = []
    for index, chunk in enumerate(chunks, start=1):
        segment = synthesizer.generate(
            chunk, ref_audio=profile.anchor_wav, ref_text=profile.anchor_text
        )
        _LOG.info(
            "第 %d/%d 段完成：%.1f 秒音频，耗时 %.1f 秒",
            index,
            len(chunks),
            segment.audio_seconds,
            segment.generation_seconds,
        )
        segments.append(segment)

    return tuple(segments)


def _diff_report(
    source: Path, profile: VoiceProfile, spoken_text: str, gate: ContentGate
) -> str:
    """差异报告的唯一目的：让人不重跑一遍就能判断是真丢了内容还是 ASR 听错了。

    所以两份文本都要全文在场——摘要在这里没有价值，人要逐字比。
    """
    missing = (
        "\n".join(f"  位置 {position}：{text}" for position, text in gate.missing)
        or "  （无定位到的缺失片段：可能是全篇细碎偏差而非整段丢失）"
    )
    normalize_note = (
        "失败——覆盖率可能因工具故障而偏低，先排除这一项再判断内容"
        if gate.asr_normalize_failed
        else "正常"
    )
    return (
        f"渲染未通过内容闸门\n"
        f"源文件：{source}\n"
        f"音色：{profile.name}\n"
        f"覆盖率：{gate.coverage:.4f}（阈值 {gate.threshold:.2f}）\n"
        f"ASR 数字归一化：{normalize_note}\n"
        f"\n缺失片段：\n{missing}\n"
        f"\n朗读文本（归一化后，模型实际收到的输入）：\n{spoken_text}\n"
        f"\nASR 原话：\n{gate.asr_text}\n"
    )


def render(
    source: Path,
    profile: VoiceProfile,
    synthesizer: SynthLike,
    transcriber: TranscriberLike,
    *,
    output: Path | None = None,
    forced: bool = False,
) -> RenderResult:
    """把一篇 markdown 渲染成带证据的音频。

    Raises:
        FileNotFoundError: 源文件不存在。
        ValueError: 清洗后无可朗读内容，或合成器采样率不符。
        BudgetError: 单段超出 token 上限（切段阈值配置有误）。
    """
    spoken_text = _prepare_text(source)
    segments = _synthesize(spoken_text, profile, synthesizer)

    audio_path = output or default_output(source)
    partial = audio_path.with_suffix(_PARTIAL_SUFFIX)
    joined = assemble.join([s.audio for s in segments], OUTPUT_SAMPLE_RATE)
    wavio.write(partial, joined, OUTPUT_SAMPLE_RATE)

    content_gate = check_content(partial, spoken_text, transcriber)
    diff_path: Path | None = None
    if not content_gate.passed:
        # 失败不在这里记日志：它已经在返回值与差异报告里了，报告方式由调用方决定。
        diff_path = audio_path.with_suffix(_DIFF_SUFFIX)
        diff_path.write_text(
            _diff_report(source, profile, spoken_text, content_gate), encoding="utf-8"
        )

    if not content_gate.passed and not forced:
        partial.unlink()
        return RenderResult(
            source=source,
            voice=profile.name,
            spoken_text=spoken_text,
            audio_path=None,
            manifest_path=None,
            diff_path=diff_path,
            content_gate=content_gate,
            voice_gate=None,
            manifest=None,
        )

    voice_gate = check_voice(
        [s.audio for s in segments], OUTPUT_SAMPLE_RATE, profile.baseline_p10_f0
    )
    manifest = assemble.build_manifest(
        source=source,
        voice=profile.name,
        spoken_text=spoken_text,
        segments=segments,
        audio_seconds=len(joined) / OUTPUT_SAMPLE_RATE,
        content_gate=content_gate,
        voice_gate=voice_gate,
        model_id=synthesizer.model_id,
        forced=forced,
        asr_model_id=ASR_MODEL_ID,
    )

    manifest_path = audio_path.with_suffix(_MANIFEST_SUFFIX)
    os.replace(partial, audio_path)
    assemble.write_manifest(manifest_path, manifest)

    return RenderResult(
        source=source,
        voice=profile.name,
        spoken_text=spoken_text,
        audio_path=audio_path,
        manifest_path=manifest_path,
        diff_path=diff_path,
        content_gate=content_gate,
        voice_gate=voice_gate,
        manifest=manifest,
    )
