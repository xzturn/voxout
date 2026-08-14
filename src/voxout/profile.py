"""音色档案：把「那个人」固定下来。

设计 §4 的实测结论是锚点为正确性要求而非可选项——零样本模式下同一段文本
两次调用的 p10 F0 极差 53.9%，等于每次换一个人说话。档案 = 锚点音频 +
锚点文本 + 基线 p10 F0，三者缺一不可：

  anchor.wav   传给模型作 ref_audio（模型内部会降到 16kHz，但我们的 LTAS
               比较要求它与输出同为 48kHz）
  anchor.txt   传给模型作 ref_text。为空会退化成非克隆模式，音色随机
  profile.toml 基线 p10 F0（闸门 2 的分母）与铸造溯源

校验从严：一个坏档案会污染此后所有对外内容，而症状（音色不对）出现得晚且
难归因。所以每条失败都给出可直接复制的修复命令。
"""

import json
import shutil
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from voxout import wavio
from voxout.config import OUTPUT_SAMPLE_RATE, VOICE_ROOT
from voxout.metrics import p10_f0

_ANCHOR_WAV = "anchor.wav"
_ANCHOR_TXT = "anchor.txt"
_PROFILE_TOML = "profile.toml"


@dataclass(frozen=True)
class VoiceProfile:
    """一个音色的全部信息。不做校验——校验是 load/mint 的边界职责。"""

    name: str
    anchor_wav: Path
    anchor_text: str
    baseline_p10_f0: float
    source: str
    seed: int | None
    mint_text: str | None
    created: str


class GeneratedAudio(Protocol):
    audio: np.ndarray
    sample_rate: int


class SynthLike(Protocol):
    """mint_profile 对合成器的全部要求。单测传替身即可，不必加载模型。"""

    def generate(self, text: str, *, seed: int | None = ...) -> GeneratedAudio: ...


def _mint_hint(name: str) -> str:
    return f"铸造命令：uv run voxout mint-voice {name}"


def _validate_anchor_geometry(anchor_wav: Path) -> None:
    info = wavio.probe(anchor_wav)
    if info.sample_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(
            f"锚点 {anchor_wav} 是 {info.sample_rate} Hz，要求 {OUTPUT_SAMPLE_RATE} Hz。"
            f"模型自己不在乎（内部会降到 16kHz），但与 48kHz 输出做 LTAS 比较时"
            f"带宽必须一致。重采样：\n"
            f"  ffmpeg -i {anchor_wav} -ar {OUTPUT_SAMPLE_RATE} -ac 1 -c:a pcm_s16le fixed.wav"
        )
    if info.channels != 1:
        raise ValueError(
            f"锚点 {anchor_wav} 是 {info.channels} 声道，要求单声道。下混：\n"
            f"  ffmpeg -i {anchor_wav} -ac 1 -ar {OUTPUT_SAMPLE_RATE} -c:a pcm_s16le fixed.wav"
        )


def _read_anchor_text(directory: Path, name: str) -> str:
    path = directory / _ANCHOR_TXT
    if not path.is_file():
        raise FileNotFoundError(f"缺少锚点文本 {path}。{_mint_hint(name)}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(
            f"{path}（anchor.txt）为空。空的 ref_text 会让模型退化成非克隆模式，"
            f"音色随机——这是静默的正确性事故，必须填入锚点音频的准确转写。"
        )
    return text


def _read_baseline(meta: dict[str, object], path: Path) -> float:
    baseline = meta.get("baseline_p10_f0")
    if not isinstance(baseline, (int, float)) or isinstance(baseline, bool) or baseline <= 0:
        raise ValueError(
            f"{path} 的 baseline_p10_f0 缺失或非正（得到 {baseline!r}）。"
            f"它是闸门 2 漂移百分比的分母，为 0 会产出 inf 或 nan。"
        )
    return float(baseline)


def load_profile(name: str, root: Path = VOICE_ROOT) -> VoiceProfile:
    """加载并全面校验一个音色档案。

    Raises:
        FileNotFoundError: 目录或必需文件缺失，消息含铸造命令。
        ValueError: 锚点格式或基线不合法，消息含 ffmpeg 修复命令。
    """
    directory = root / name
    if not directory.is_dir():
        raise FileNotFoundError(f"音色 {name!r} 不存在：{directory}。{_mint_hint(name)}")

    anchor_wav = directory / _ANCHOR_WAV
    if not anchor_wav.is_file():
        raise FileNotFoundError(f"缺少锚点音频 {anchor_wav}。{_mint_hint(name)}")

    toml_path = directory / _PROFILE_TOML
    if not toml_path.is_file():
        raise FileNotFoundError(f"缺少档案元数据 {toml_path}。{_mint_hint(name)}")

    _validate_anchor_geometry(anchor_wav)
    anchor_text = _read_anchor_text(directory, name)
    meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))

    return VoiceProfile(
        name=name,
        anchor_wav=anchor_wav,
        anchor_text=anchor_text,
        baseline_p10_f0=_read_baseline(meta, toml_path),
        source=str(meta.get("source", "unknown")),
        seed=meta.get("seed") if isinstance(meta.get("seed"), int) else None,
        mint_text=meta.get("mint_text") if isinstance(meta.get("mint_text"), str) else None,
        created=str(meta.get("created", "")),
    )


def _render_toml(profile: VoiceProfile) -> str:
    """手拼 TOML（标准库无 writer，字段少）。

    字符串值用 json.dumps 转义：JSON 字符串转义规则是 TOML 基本字符串的子集，
    对我们的内容（中文 + 常见标点）完全等价，且不必自己处理引号与反斜杠。
    """
    lines = [
        f"name = {json.dumps(profile.name, ensure_ascii=False)}",
        f"source = {json.dumps(profile.source, ensure_ascii=False)}",
        f"baseline_p10_f0 = {profile.baseline_p10_f0!r}",
        f"created = {json.dumps(profile.created, ensure_ascii=False)}",
    ]
    if profile.seed is not None:
        lines.append(f"seed = {profile.seed}")
    if profile.mint_text is not None:
        lines.append(f"mint_text = {json.dumps(profile.mint_text, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def mint_profile(
    name: str,
    text: str,
    synthesizer: SynthLike,
    *,
    seed: int | None = None,
    root: Path = VOICE_ROOT,
) -> VoiceProfile:
    """零样本生成一段音频，把它固定成此后所有渲染的音色锚点。

    拒绝覆盖已有档案：覆盖 house 音色会静默改变此后每一次对外输出的声音，
    这个动作必须由人显式删除目录来表达。

    任何一步失败都会清掉半成品目录——半个档案比没有档案更糟，它会通过
    一部分校验然后在别处炸。

    Raises:
        FileExistsError: 档案已存在。
        ValueError: 文本为空、采样率不符，或生成的音频测不出浊音基频。
    """
    if not text.strip():
        raise ValueError("锚点文本为空，无法铸造")

    directory = root / name
    if directory.exists():
        raise FileExistsError(
            f"音色 {name!r} 已存在：{directory}。覆盖会静默改变此后所有输出的声音；"
            f"确认要重铸请先显式删除该目录。"
        )

    generated = synthesizer.generate(text, seed=seed)
    if generated.sample_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(
            f"合成器输出 {generated.sample_rate} Hz，要求 {OUTPUT_SAMPLE_RATE} Hz"
        )

    audio = np.asarray(generated.audio, dtype=np.float32)
    baseline = p10_f0(audio, generated.sample_rate)
    if not np.isfinite(baseline) or baseline <= 0:
        raise ValueError(
            "生成的音频里没有可用的浊音帧，测不出 p10 F0。这样的锚点会让闸门 2 "
            "永久比对 nan。请检查锚点文本是否可朗读，或换一个种子重试。"
        )

    profile = VoiceProfile(
        name=name,
        anchor_wav=directory / _ANCHOR_WAV,
        anchor_text=text.strip(),
        baseline_p10_f0=baseline,
        source="minted",
        seed=seed,
        mint_text=text.strip(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    try:
        directory.mkdir(parents=True)
        wavio.write(profile.anchor_wav, audio, generated.sample_rate)
        (directory / _ANCHOR_TXT).write_text(profile.anchor_text, encoding="utf-8")
        (directory / _PROFILE_TOML).write_text(_render_toml(profile), encoding="utf-8")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    return profile
