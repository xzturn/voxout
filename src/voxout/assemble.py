"""把多段音频拼成一个文件，并记录这次渲染的全部可审计信息。

manifest 不是日志，是**证据**。闸门读数只在特定库版本、特定模型下可比，而
「上次是 98.9%，这次 96.2%」这种问题只有在版本、种子、文本哈希都留在文件里
时才答得出。所以模型 id、库版本、文本哈希、是否 --force 绕过都进常规输出。

两个容易安静出错的地方，在这里被一次性解决：

  nan       `json.dumps(float("nan"))` 写出的 `NaN` 不是合法 JSON。而 nan
            恰恰出现在闸门失败时——最需要 manifest 的时候。
  numpy 标量 `json.dumps` 遇到它直接 TypeError。且 `np.float64` 是 Python
            `float` 的子类，只判 `isinstance(x, float)` 抓不住它。
"""

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from voxout.config import ASR_MODEL_ID, CHUNK_GAP_SECONDS
from voxout.synth import Segment
from voxout.textmatch import han_only
from voxout.verify import ContentGate, VoiceGate

_LOG = logging.getLogger(__name__)

_TRACKED_DISTRIBUTIONS = {"mlx_audio": "mlx-audio", "mlx": "mlx", "voxout": "voxout"}


def join(
    segments: Sequence[np.ndarray], sr: int, gap_seconds: float = CHUNK_GAP_SECONDS
) -> np.ndarray:
    """按顺序拼接各段，段间插入静音。

    间隔是**分隔符而非后缀**：n 段插 n-1 段静音。末尾多一段静音会让成品结尾拖沓。

    总是返回新数组——原段还要拿去测 p10 F0（闸门 2），不能让调用方通过结果
    间接改到它们。

    Raises:
        ValueError: 无段可拼、某段非一维或为空、间隔为负、采样率非正。
    """
    if sr <= 0:
        raise ValueError(f"采样率必须为正，收到 {sr}")
    if gap_seconds < 0:
        raise ValueError(f"段间间隔不能为负，收到 {gap_seconds}")
    if len(segments) == 0:
        raise ValueError("没有音频段可拼接")

    prepared = [_validated(segment, index) for index, segment in enumerate(segments)]
    if len(prepared) == 1:
        return prepared[0].copy()

    gap = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    pieces: list[np.ndarray] = [prepared[0]]
    for array in prepared[1:]:
        pieces.extend((gap, array))
    return np.concatenate(pieces)


def _validated(segment: np.ndarray, index: int) -> np.ndarray:
    array = np.asarray(segment, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"第 {index + 1} 段不是一维信号（shape={array.shape}）")
    if array.size == 0:
        raise ValueError(
            f"第 {index + 1} 段为空。说明上游某次生成没有产出，"
            f"静默拼进去会让 manifest 的时长与实际对不上。"
        )
    return array


def _json_safe(value: object) -> object:
    """递归清洗成严格 JSON 可写的形式。

    numpy 标量的判断必须排在 float 之前：`np.float64` 是 `float` 的子类，
    顺序反了的话有限的 numpy 浮点会原样漏过去，然后在 `json.dumps` 抛 TypeError。
    """
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _library_versions() -> dict[str, str]:
    """记录库版本。取不到时记 unknown 并告警——manifest 是诊断文件，
    不该因为一个版本字符串缺失而让整次渲染失败。"""
    versions: dict[str, str] = {}
    for name, distribution in _TRACKED_DISTRIBUTIONS.items():
        try:
            versions[name] = version(distribution)
        except PackageNotFoundError:
            _LOG.warning("取不到 %s 的版本号，manifest 记为 unknown", distribution)
            versions[name] = "unknown"
    return versions


def build_manifest(
    *,
    source: Path,
    voice: str,
    spoken_text: str,
    segments: Sequence[Segment],
    audio_seconds: float,
    content_gate: ContentGate,
    voice_gate: VoiceGate,
    model_id: str,
    forced: bool,
    asr_model_id: str = ASR_MODEL_ID,
) -> dict[str, object]:
    """汇总一次渲染的全部可审计信息，返回严格 JSON 可写的字典。

    `speech_seconds` 与 `audio_seconds` 有意分开：前者是各段语音时长之和，
    后者是成品文件时长（含拼接插入的静音）。`seconds_per_han` 用前者——
    它存在的意义是与 budget 里 0.30 秒/汉字的估计相互印证（设计 §6），
    掺了静音就随段数漂移，不再可比，也就不再是旁证。

    Raises:
        ValueError: 没有音频段。
    """
    if len(segments) == 0:
        raise ValueError("没有音频段，无从生成 manifest")

    han_chars = len(han_only(spoken_text))
    speech_seconds = sum(segment.audio_seconds for segment in segments)

    manifest = {
        "input": source,
        "voice": voice,
        "text_sha256": hashlib.sha256(spoken_text.encode("utf-8")).hexdigest(),
        "chars": len(spoken_text),
        "han_chars": han_chars,
        "segments": len(segments),
        "audio_seconds": audio_seconds,
        "speech_seconds": speech_seconds,
        "generation_seconds": sum(s.generation_seconds for s in segments),
        "seconds_per_han": speech_seconds / han_chars if han_chars else None,
        "max_tokens": sum(s.max_tokens for s in segments),
        "token_count": sum(s.token_count for s in segments),
        "peak_memory_gb": max(s.peak_memory_gb for s in segments),
        "gate_content": asdict(content_gate),
        "gate_voice": asdict(voice_gate),
        "model_id": model_id,
        "asr_model_id": asr_model_id,
        "versions": _library_versions(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "forced": forced,
    }
    return _json_safe(manifest)  # type: ignore[return-value]


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """写出 manifest。

    `allow_nan=False` 是自检而非风格选择：漏掉任何一个非有限值会在这里抛错，
    而不是写出一个下游解析器全都拒收的文件。
    `ensure_ascii=False` 同理——manifest 是给人看的，`\\u7b2c` 等于没写。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
