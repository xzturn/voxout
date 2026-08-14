"""音色档案：把「那个人」固定下来。

设计 §4 的实测结论是锚点为正确性要求而非可选项——零样本模式下同一段文本
两次调用的 p10 F0 极差 53.9%，等于每次换一个人说话。档案 = 锚点音频 +
锚点文本 + 基线 p10 F0，三者缺一不可：

  anchor.wav   传给模型作 ref_audio（模型内部会降到 16kHz，但我们的 LTAS
               比较要求它与输出同为 48kHz）
  anchor.txt   传给模型作 ref_text。为空会退化成非克隆模式，音色随机
  profile.toml 基线 p10 F0（闸门 2 的分母）与铸造溯源。基线不是锚点自身的读数，
               而是**锚点条件下**若干次生成的中位数——理由见 `_measure_baseline`

校验从严：一个坏档案会污染此后所有对外内容，而症状（音色不对）出现得晚且
难归因。所以每条失败都给出可直接复制的修复命令。
"""

import json
import logging
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from voxout import wavio
from voxout.config import BASELINE_SAMPLES, OUTPUT_SAMPLE_RATE, VOICE_ROOT
from voxout.metrics import p10_f0

_LOG = logging.getLogger(__name__)

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

    def generate(
        self,
        text: str,
        *,
        ref_audio: Path | None = ...,
        ref_text: str | None = ...,
        seed: int | None = ...,
    ) -> GeneratedAudio: ...


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


def _audio_of(generated: GeneratedAudio) -> np.ndarray:
    """取出音频，同时钉住采样率。

    按错的采样率解读波形会让 F0 系统性偏移，表现为「音色漂移」而真实原因是单位错。

    Raises:
        ValueError: 采样率不是 48kHz。
    """
    if generated.sample_rate != OUTPUT_SAMPLE_RATE:
        raise ValueError(
            f"合成器输出 {generated.sample_rate} Hz，要求 {OUTPUT_SAMPLE_RATE} Hz"
        )
    return np.asarray(generated.audio, dtype=np.float32)


def _require_voiced_p10(audio: np.ndarray, what: str) -> float:
    """测 p10 F0，并拒绝测不出来的音频。

    Raises:
        ValueError: 没有可用的浊音帧。
    """
    value = p10_f0(audio, OUTPUT_SAMPLE_RATE)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{what}里没有可用的浊音帧，测不出 p10 F0。这样的锚点会让闸门 2 "
            f"永久比对 nan。请检查锚点文本是否可朗读，或换一个种子重试。"
        )
    return float(value)


def _sample_seed(seed: int | None, index: int) -> int | None:
    """第 index 份条件采样的种子。

    绝不共用锚点那个种子：同种子同条件下模型是可复现的，N 份读数会完全相同，
    中位数于是悄悄退化成 n=1——一个「看起来是 N 份采样」的 1 份采样。没给种子
    就一份都不钉，从 None 派生出数字只会假装可复现。
    """
    return None if seed is None else seed + 1 + index


def _measure_baseline(
    anchor_wav: Path,
    anchor_text: str,
    synthesizer: SynthLike,
    seed: int | None,
) -> float:
    """基线 = `BASELINE_SAMPLES` 次**锚点条件下**生成的 p10 F0 中位数。

    量条件生成而不是锚点自身，是因为闸门 2 在渲染时量的就是条件生成，拿锚点自身
    当分母属于跨分布比较。但这一条是原则，不是在修一个观测到的偏差：实测两者只
    差 0.31%（166.32 → 165.81），它解释不了真实内容里 23 段有 22 段漂移为负、
    中位数 −5.41%。

    真正的缺陷是 n=1。文本与说话人固定、只换采样，p10 F0 的极差实测 7.71% 与
    14.91%，单次读数落在一个约 ±15% 宽的分布里，而基线是此后**每一次**判定的
    分母。上面那 −5.41% 也在这个宽度之内。

    取中位数而不是均值：单个离群读数（八度错误、某次生成含大量清音）会把均值
    整体拖走，被拖走一次就长期偏。

    各份读数与极差写进日志（CLI 已配 INFO）。中位数本身不携带离散度，而极差就是
    闸门 2 的噪声下限——同一个中位数，极差 1% 与 30% 对阈值的含义完全不同，而铸造
    结束后没人能再重建这几份读数。
    """
    readings = tuple(
        _require_voiced_p10(
            _audio_of(
                synthesizer.generate(
                    anchor_text,
                    ref_audio=anchor_wav,
                    ref_text=anchor_text,
                    seed=_sample_seed(seed, index),
                )
            ),
            f"第 {index + 1} 份条件采样",
        )
        for index in range(BASELINE_SAMPLES)
    )
    baseline = float(np.median(readings))
    _LOG.info(
        "基线 %.2f Hz = %d 份条件采样的中位数；各份 %s Hz，极差 %.2f%%（闸门 2 的噪声下限）",
        baseline,
        BASELINE_SAMPLES,
        " / ".join(f"{value:.1f}" for value in readings),
        (max(readings) - min(readings)) / baseline * 100.0,
    )
    return baseline


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

    锚点先写进临时目录，因为定基线要拿它当参考音再生成几次（`_measure_baseline`）。
    等基线测出来才建正式目录——正式目录里于是永远不会出现半个档案，这个性质是
    结构性的，不依赖异常清理路径跑到。

    拒绝覆盖已有档案：覆盖 house 音色会静默改变此后每一次对外输出的声音，
    这个动作必须由人显式删除目录来表达。

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

    anchor_text = text.strip()
    audio = _audio_of(synthesizer.generate(anchor_text, seed=seed))
    # 锚点自身也要能测出基频：测不出说明这段音频根本不可用作参考音，
    # 此时就该失败，而不是先花几次生成去测一个注定无意义的基线。
    _require_voiced_p10(audio, "生成的锚点音频")

    with tempfile.TemporaryDirectory(prefix="voxout-mint-") as staging:
        staging_wav = Path(staging) / _ANCHOR_WAV
        wavio.write(staging_wav, audio, OUTPUT_SAMPLE_RATE)
        baseline = _measure_baseline(staging_wav, anchor_text, synthesizer, seed)

    profile = VoiceProfile(
        name=name,
        anchor_wav=directory / _ANCHOR_WAV,
        anchor_text=anchor_text,
        baseline_p10_f0=baseline,
        source="minted",
        seed=seed,
        mint_text=anchor_text,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    try:
        directory.mkdir(parents=True)
        wavio.write(profile.anchor_wav, audio, OUTPUT_SAMPLE_RATE)
        (directory / _ANCHOR_TXT).write_text(profile.anchor_text, encoding="utf-8")
        (directory / _PROFILE_TOML).write_text(_render_toml(profile), encoding="utf-8")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    return profile
