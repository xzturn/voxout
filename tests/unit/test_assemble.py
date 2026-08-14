"""拼接与 manifest。

manifest 的测试重点不是「字段齐全」，而是**它真的能被写成合法 JSON**。
nan 和 numpy 标量是两个安静的雷：`json.dumps(float("nan"))` 会写出 `NaN`，
那不是合法 JSON，任何下游解析器都会拒收；而 numpy 标量在 `json.dumps` 下
直接抛 TypeError。两者都只在闸门失败或某段静音时才出现——也就是最需要
manifest 的时候。
"""

import json
import logging
import math
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

from voxout import assemble
from voxout.assemble import build_manifest, join, write_manifest
from voxout.config import CHUNK_GAP_SECONDS, OUTPUT_SAMPLE_RATE
from voxout.synth import Segment
from voxout.verify import ContentGate, VoiceGate

SR = OUTPUT_SAMPLE_RATE


def ramp(n: int, start: float = 0.1) -> np.ndarray:
    """非零、可区分的测试信号——全零无法验证间隔是否插在正确位置。"""
    return np.full(n, start, dtype=np.float32)


def make_segment(
    audio: np.ndarray,
    *,
    generation_seconds: float = 2.0,
    token_count: int = 100,
    max_tokens: int = 200,
    peak_memory_gb: float = 3.0,
) -> Segment:
    return Segment(
        audio=audio,
        sample_rate=SR,
        audio_seconds=len(audio) / SR,
        generation_seconds=generation_seconds,
        token_count=token_count,
        max_tokens=max_tokens,
        peak_memory_gb=peak_memory_gb,
    )


def make_content_gate(coverage: float = 0.99) -> ContentGate:
    return ContentGate(
        coverage=coverage,
        threshold=0.97,
        passed=coverage >= 0.97,
        asr_text="听到的文本",
        missing=(),
        asr_normalize_failed=False,
    )


def make_voice_gate(drift: float = 1.5) -> VoiceGate:
    return VoiceGate(
        baseline_p10=180.0,
        segment_p10s=(180.0, 182.0),
        drift_pct=drift,
        spread_pct=1.1,
        threshold=6.0,
        passed=drift <= 6.0,
    )


class TestJoin:
    def test_single_segment_passes_through_unchanged(self) -> None:
        audio = ramp(1000)

        joined = join([audio], SR)

        assert np.array_equal(joined, audio)

    def test_two_segments_are_separated_by_exactly_one_gap(self) -> None:
        first, second = ramp(1000, 0.1), ramp(2000, 0.2)
        gap_samples = int(SR * CHUNK_GAP_SECONDS)

        joined = join([first, second], SR)

        assert len(joined) == 1000 + gap_samples + 2000

    def test_the_gap_is_actually_silence_and_sits_between_the_segments(self) -> None:
        """长度对但间隔插错位置的话，听起来是句中断句——所以要验位置，不只验长度。"""
        first, second = ramp(1000, 0.1), ramp(2000, 0.2)
        gap_samples = int(SR * CHUNK_GAP_SECONDS)

        joined = join([first, second], SR)

        assert np.all(joined[:1000] == 0.1)
        assert np.all(joined[1000 : 1000 + gap_samples] == 0.0)
        assert np.all(joined[1000 + gap_samples :] == 0.2)

    def test_zero_gap_gives_exact_concatenation(self) -> None:
        first, second = ramp(10, 0.1), ramp(20, 0.2)

        joined = join([first, second], SR, gap_seconds=0.0)

        assert len(joined) == 30

    def test_three_segments_get_two_gaps_not_three(self) -> None:
        """间隔是分隔符而非后缀：末尾多一段静音会让成品结尾拖沓。"""
        gap_samples = int(SR * CHUNK_GAP_SECONDS)

        joined = join([ramp(100), ramp(100), ramp(100)], SR)

        assert len(joined) == 300 + 2 * gap_samples

    def test_inputs_are_never_mutated(self) -> None:
        """拼接必须返回新数组。原段还要拿去测 p10 F0（闸门 2）。"""
        first, second = ramp(100, 0.1), ramp(100, 0.2)
        before = first.copy()

        join([first, second], SR)

        assert np.array_equal(first, before)

    def test_result_is_float32(self) -> None:
        assert join([ramp(100), ramp(100)], SR).dtype == np.float32

    def test_no_segments_is_refused(self) -> None:
        with pytest.raises(ValueError, match="没有"):
            join([], SR)

    def test_multichannel_segment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="一维"):
            join([np.zeros((2, 100), dtype=np.float32)], SR)

    def test_empty_segment_is_refused(self) -> None:
        """空段说明上游某次生成没有产出，静默拼进去会让 manifest 的时长对不上。"""
        with pytest.raises(ValueError, match="空"):
            join([ramp(100), np.array([], dtype=np.float32)], SR)

    def test_negative_gap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="间隔"):
            join([ramp(100), ramp(100)], SR, gap_seconds=-0.1)

    def test_non_positive_sample_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="采样率"):
            join([ramp(100)], 0)


class TestBuildManifest:
    def manifest(self, **overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "source": Path("posts/example.md"),
            "voice": "house",
            "spoken_text": "第一段内容。第二段内容。",
            "segments": [make_segment(ramp(SR)), make_segment(ramp(2 * SR))],
            "audio_seconds": 3.2,
            "content_gate": make_content_gate(),
            "voice_gate": make_voice_gate(),
            "model_id": "mlx-community/VoxCPM2-8bit",
            "forced": False,
        }
        return build_manifest(**{**defaults, **overrides})  # type: ignore[arg-type]

    def test_all_documented_fields_are_present(self) -> None:
        expected = {
            "input", "voice", "text_sha256", "chars", "han_chars", "segments",
            "audio_seconds", "speech_seconds", "generation_seconds", "seconds_per_han",
            "max_tokens", "token_count", "peak_memory_gb", "gate_content", "gate_voice",
            "model_id", "asr_model_id", "versions", "created", "forced",
        }  # fmt: skip

        assert set(self.manifest()) == expected

    def test_serializes_to_strict_json(self) -> None:
        """allow_nan=False 是自检：漏掉任何一个非有限值都会在这里抛错而非写出坏文件。"""
        text = json.dumps(self.manifest(), ensure_ascii=False, allow_nan=False)

        assert "house" in text

    def test_path_becomes_a_string(self) -> None:
        assert self.manifest()["input"] == "posts/example.md"

    def test_counts_come_from_the_spoken_text(self) -> None:
        manifest = self.manifest(spoken_text="第一段内容。第二段内容。")

        assert manifest["chars"] == 12
        assert manifest["han_chars"] == 10
        assert manifest["segments"] == 2

    def test_rates_and_totals_are_aggregated_correctly(self) -> None:
        manifest = self.manifest()

        assert manifest["speech_seconds"] == pytest.approx(3.0)
        assert manifest["generation_seconds"] == pytest.approx(4.0)
        assert manifest["token_count"] == 200
        assert manifest["max_tokens"] == 400
        assert manifest["peak_memory_gb"] == pytest.approx(3.0)

    def test_seconds_per_han_uses_speech_time_not_file_duration(self) -> None:
        """必须用语音时长，不能用文件时长。

        文件时长含拼接插入的静音，段数一多速率就被稀释——而这个数存在的意义
        是与 budget 里 0.30 秒/汉字的估计相互印证（设计 §6）。掺了静音就不再可比。
        """
        manifest = self.manifest(audio_seconds=999.0)

        assert manifest["seconds_per_han"] == pytest.approx(3.0 / 10)
        assert manifest["audio_seconds"] == pytest.approx(999.0)

    def test_seconds_per_han_is_null_when_there_are_no_han_characters(self) -> None:
        manifest = self.manifest(spoken_text="abc")

        assert manifest["seconds_per_han"] is None

    def test_text_sha256_identifies_the_synthesized_text(self) -> None:
        """哈希的是**朗读文本**而非源 markdown：那才是模型实际收到的输入。"""
        import hashlib

        text = "第一段内容。"
        manifest = self.manifest(spoken_text=text)

        assert manifest["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()

    def test_gates_are_embedded_in_full(self) -> None:
        gates = self.manifest()
        content = gates["gate_content"]
        assert isinstance(content, dict)
        assert content["coverage"] == pytest.approx(0.99)
        assert content["passed"] is True

    def test_non_finite_gate_values_become_null(self) -> None:
        """静音段会让漂移为 nan。`NaN` 不是合法 JSON，必须落成 null。"""
        manifest = self.manifest(voice_gate=make_voice_gate(drift=math.nan))
        voice = manifest["gate_voice"]

        assert isinstance(voice, dict)
        assert voice["drift_pct"] is None
        assert voice["passed"] is False
        json.dumps(manifest, allow_nan=False)

    def test_numpy_scalars_never_leak_through(self) -> None:
        """numpy 标量在 json.dumps 下直接抛 TypeError，且 np.float64 是 float 的子类，
        只判 isinstance(x, float) 是抓不住它的。"""
        manifest = self.manifest(
            voice_gate=VoiceGate(
                baseline_p10=np.float64(180.0),
                segment_p10s=(np.float64(180.0),),
                drift_pct=np.float64(1.5),
                spread_pct=np.float32(0.0),
                threshold=6.0,
                passed=np.bool_(True),
            )
        )
        voice = manifest["gate_voice"]

        assert isinstance(voice, dict)
        assert type(voice["drift_pct"]) is float
        assert type(voice["passed"]) is bool
        json.dumps(manifest, allow_nan=False)

    def test_library_versions_are_recorded(self) -> None:
        """闸门读数只在特定库版本下可比。版本不进 manifest，标定就无从追溯。"""
        versions = self.manifest()["versions"]

        assert isinstance(versions, dict)
        assert set(versions) == {"mlx_audio", "mlx", "voxout"}
        assert all(isinstance(v, str) for v in versions.values())

    def test_unavailable_version_becomes_unknown_rather_than_failing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """manifest 是诊断文件。一个取不到的版本号不该让整次渲染（几分钟的生成）作废。

        但必须留痕：静默记成空串会让人以为版本确实是空的。
        """

        def missing(_distribution: str) -> str:
            raise PackageNotFoundError("模拟未安装")

        monkeypatch.setattr(assemble, "version", missing)

        with caplog.at_level(logging.WARNING):
            versions = self.manifest()["versions"]

        assert isinstance(versions, dict)
        assert set(versions.values()) == {"unknown"}
        assert "unknown" in caplog.text

    def test_forced_flag_is_recorded(self) -> None:
        """--force 绕过闸门。哪次产物是绕过来的，必须留在文件里。"""
        assert self.manifest(forced=True)["forced"] is True

    def test_no_segments_is_refused(self) -> None:
        with pytest.raises(ValueError, match="没有"):
            self.manifest(segments=[])


class TestWriteManifest:
    def test_round_trips_through_json(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "manifest.json"
        manifest = {"voice": "house", "coverage": 0.99, "missing": [[10, "第三段"]]}

        write_manifest(path, manifest)

        assert json.loads(path.read_text(encoding="utf-8")) == manifest

    def test_chinese_is_written_readable_not_escaped(self, tmp_path: Path) -> None:
        """manifest 是给人看的诊断文件。\\u7b2c 这种转义等于没写。"""
        path = tmp_path / "manifest.json"

        write_manifest(path, {"voice": "第一段"})

        assert "第一段" in path.read_text(encoding="utf-8")

    def test_non_finite_values_are_refused_rather_than_written(
        self, tmp_path: Path
    ) -> None:
        """兜底：build_manifest 已清洗过，但手工构造的 dict 不能绕过这道检查。"""
        with pytest.raises(ValueError):
            write_manifest(tmp_path / "manifest.json", {"drift": math.nan})
