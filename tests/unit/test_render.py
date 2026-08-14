"""渲染流水线：md → 朗读文本 → 切段 → 逐段生成 → 拼接 → 两道闸门 → 落盘。

这些测试全程不加载模型。流水线的价值不在「会调用模型」，而在**接线正确**：
参考音有没有传给每一段、闸门失败时音频到底出不出、失败证据有没有落盘。
每一条都是静默失效的入口。
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fakes import P10_OF_200HZ, FakeSynthesizer, RecordingTranscriber, sine

from voxout import markdown, normalize, wavio
from voxout.config import CHUNK_GAP_SECONDS, OUTPUT_SAMPLE_RATE
from voxout.profile import VoiceProfile
from voxout.render import render

SPOKEN = "第一段内容。第二段内容。"
MARKDOWN = f"""---
title: 不该被读出来的前言
---

# 标题文字要读，井号不要

{SPOKEN}

```python
print("代码块不该被读出来")
```
"""


MULTI_PARAGRAPH_MARKDOWN = """# 标题文字要读

第一自然段的内容。

第二自然段的内容。
"""
"""标题 + 两个自然段 → 三次生成。

多段不再靠文本长度制造，靠段落数：段落是原子单位（chunker 模块文档，T13 实测）。
这个夹具因此同时钉住 markdown 清洗必须把空行折成 `\\n` 而不是抹平——段落界一丢，
切段就退回 T13 那次跳读失败的行为。
"""


def spoken_of(md: str) -> str:
    """流水线会送进模型的文本。"""
    return normalize.to_speakable_numbers(markdown.to_speakable(md))


def perfect_ear(md: str = MARKDOWN) -> RecordingTranscriber:
    """一个「听全了」的 ASR：回的正是流水线送进模型的那段文本。

    用流水线自己算出期望值，是因为这个替身要表达的就是「完美转写」这个概念，
    而不是某个写死的字符串——标题文字该不该读是另外几条测试的事。
    """
    return RecordingTranscriber(spoken_of(md))


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认输出路径是相对的（`out/…`），所以每条测试都必须在自己的临时目录里跑。

    少了这个，不带 `-o` 的测试会把产物写进仓库真正的 out/——那里可能正躺着
    一次真实渲染的成品。
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def anchor(tmp_path: Path) -> Path:
    path = tmp_path / "voice" / "house" / "anchor.wav"
    wavio.write(path, sine(200.0), OUTPUT_SAMPLE_RATE)
    return path


@pytest.fixture
def profile(anchor: Path) -> VoiceProfile:
    return VoiceProfile(
        name="house",
        anchor_wav=anchor,
        anchor_text="这是一段用来测试音色是否稳定的中文句子。",
        baseline_p10_f0=P10_OF_200HZ,
        source="minted",
        seed=42,
        mint_text="这是一段用来测试音色是否稳定的中文句子。",
        created="2026-08-13T00:00:00+00:00",
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "posts" / "示例 帖子.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MARKDOWN, encoding="utf-8")
    return path


class TestHappyPath:
    def test_writes_audio_and_manifest_side_by_side(
        self, source: Path, profile: VoiceProfile, tmp_path: Path
    ) -> None:
        output = tmp_path / "out" / "示例.wav"

        result = render(
            source,
            profile,
            FakeSynthesizer(),
            perfect_ear(),
            output=output,
        )

        assert result.audio_path == output
        assert output.is_file()
        assert result.manifest_path == tmp_path / "out" / "示例.manifest.json"
        assert result.manifest_path.is_file()
        assert result.diff_path is None

    def test_both_gates_pass(self, source: Path, profile: VoiceProfile) -> None:
        result = render(source, profile, FakeSynthesizer(), perfect_ear())

        assert result.content_gate.passed
        assert result.voice_gate is not None
        assert result.voice_gate.passed
        assert result.voice_gate.drift_pct == pytest.approx(0.0, abs=1e-9)

    def test_audio_is_48k_mono_and_matches_the_manifest_duration(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """manifest 里的时长必须与文件真实时长一致，否则它作为证据没有意义。"""
        result = render(source, profile, FakeSynthesizer(), perfect_ear())

        assert result.audio_path is not None
        info = wavio.probe(result.audio_path)
        assert info.sample_rate == OUTPUT_SAMPLE_RATE
        assert info.channels == 1
        assert result.manifest is not None
        assert info.duration_seconds == pytest.approx(
            result.manifest["audio_seconds"], abs=1e-3
        )

    def test_no_partial_file_is_left_behind(
        self, source: Path, profile: VoiceProfile, tmp_path: Path
    ) -> None:
        """闸门 1 要读文件才能转写，所以音频先落临时文件。它不能留在成品目录里。"""
        output = tmp_path / "out" / "示例.wav"

        render(
            source, profile, FakeSynthesizer(), perfect_ear(), output=output
        )

        assert sorted(p.name for p in output.parent.iterdir()) == [
            "示例.manifest.json",
            "示例.wav",
        ]

    def test_manifest_is_valid_json_with_the_spoken_text_hash(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        import hashlib

        result = render(source, profile, FakeSynthesizer(), perfect_ear())

        assert result.manifest_path is not None
        written = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert written["text_sha256"] == hashlib.sha256(
            result.spoken_text.encode()
        ).hexdigest()
        assert written["voice"] == "house"
        assert written["forced"] is False
        assert written["input"] == str(source)


class TestTextPreparation:
    def test_markdown_syntax_never_reaches_the_model(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """前言、代码块、标题井号都不该被读出来——它们会被 ASR 听成噪声，
        然后表现为内容闸门莫名其妙地不过。"""
        synthesizer = FakeSynthesizer()

        result = render(source, profile, synthesizer, perfect_ear())

        spoken = result.spoken_text
        assert "---" not in spoken
        assert "```" not in spoken
        assert "print" not in spoken
        assert "#" not in spoken
        assert SPOKEN in spoken

    def test_numbers_are_normalized_before_synthesis(
        self, tmp_path: Path, profile: VoiceProfile
    ) -> None:
        """模型读阿拉伯数字不可靠。归一化必须发生在送进模型之前，
        而不是只在闸门比较时才做。"""
        path = tmp_path / "数字.md"
        path.write_text("用时 1200 毫秒。", encoding="utf-8")
        synthesizer = FakeSynthesizer()

        render(path, profile, synthesizer, RecordingTranscriber("用时1200毫秒。"))

        sent = str(synthesizer.calls[0]["text"])
        assert "1200" not in sent
        assert "一千二百" in sent

    def test_each_paragraph_becomes_its_own_generation_with_gaps_between_them(
        self, tmp_path: Path, profile: VoiceProfile
    ) -> None:
        """一次生成一个段落，n 段之间留 n-1 段静音。

        段数在这里是写死的 3（标题 + 两个自然段），因为「段落是原子单位」这条
        结论本身就是一个计数断言：合并任意两段都会让它变成 2。T13 实测里 213 汉字
        单段送出的覆盖率是 55.40%，切到段落级是 99.06%。
        """
        path = tmp_path / "多段.md"
        path.write_text(MULTI_PARAGRAPH_MARKDOWN, encoding="utf-8")
        synthesizer = FakeSynthesizer()

        result = render(path, profile, synthesizer, perfect_ear(MULTI_PARAGRAPH_MARKDOWN))

        assert len(synthesizer.calls) == 3
        assert result.manifest is not None
        assert result.manifest["segments"] == 3
        gaps = 2 * CHUNK_GAP_SECONDS
        assert result.manifest["audio_seconds"] == pytest.approx(
            result.manifest["speech_seconds"] + gaps, abs=1e-3
        )


class TestAnchorWiring:
    def test_every_segment_gets_the_anchor_audio_and_text(
        self, tmp_path: Path, profile: VoiceProfile
    ) -> None:
        """设计 §4 的核心正确性要求：任何一段漏掉锚点，那一段就换了个人说话。

        零样本模式下同一段文本两次调用的 p10 F0 极差 53.9%——远超闸门 2 的
        6% 余量，但闸门只看漂移不看**为什么**漂移，所以这条必须在这里钉住。
        """
        path = tmp_path / "多段.md"
        path.write_text(MULTI_PARAGRAPH_MARKDOWN, encoding="utf-8")
        synthesizer = FakeSynthesizer()

        render(path, profile, synthesizer, perfect_ear(MULTI_PARAGRAPH_MARKDOWN))

        assert len(synthesizer.calls) > 1, "多段才谈得上「每段都带锚点」"
        for call in synthesizer.calls:
            assert call["ref_audio"] == profile.anchor_wav
            assert call["ref_text"] == profile.anchor_text

    def test_voice_gate_is_measured_against_the_profile_baseline(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        result = render(source, profile, FakeSynthesizer(), perfect_ear())

        assert result.voice_gate is not None
        assert result.voice_gate.baseline_p10 == profile.baseline_p10_f0


class TestContentGateFailure:
    def test_no_audio_is_written_and_the_evidence_is(
        self, source: Path, profile: VoiceProfile, tmp_path: Path
    ) -> None:
        """设计 §9：闸门 1 不过 = 渲染失败，写差异报告，不出音频。

        不出音频是对的——一个丢了内容的音频文件比没有文件危险，它看起来像成品。
        """
        output = tmp_path / "out" / "示例.wav"

        result = render(
            source,
            profile,
            FakeSynthesizer(),
            RecordingTranscriber("第一段内容。"),
            output=output,
        )

        assert not result.content_gate.passed
        assert result.audio_path is None
        assert result.manifest_path is None
        assert not output.is_file()
        assert result.diff_path == tmp_path / "out" / "示例.diff.txt"
        assert result.diff_path.is_file()

    def test_the_partial_audio_is_cleaned_up(
        self, source: Path, profile: VoiceProfile, tmp_path: Path
    ) -> None:
        output = tmp_path / "out" / "示例.wav"

        render(
            source,
            profile,
            FakeSynthesizer(),
            RecordingTranscriber("第一段内容。"),
            output=output,
        )

        assert [p.name for p in output.parent.iterdir()] == ["示例.diff.txt"]

    def test_the_report_carries_both_texts_and_the_missing_spans(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """报告要让人在不重跑一遍的前提下判断「是真丢了内容，还是 ASR 听错了」。
        为此两份文本都要在，缺失位置也要在。"""
        result = render(
            source, profile, FakeSynthesizer(), RecordingTranscriber("第一段内容。")
        )

        assert result.diff_path is not None
        report = result.diff_path.read_text(encoding="utf-8")
        assert "第二段内容" in report
        assert "第一段内容。" in report
        assert f"{result.content_gate.coverage:.4f}" in report
        assert "0.97" in report

    def test_voice_gate_is_not_measured_when_content_already_failed(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """内容错了，音色读数没有意义，也不该让人以为有一半通过了。"""
        result = render(
            source, profile, FakeSynthesizer(), RecordingTranscriber("第一段内容。")
        )

        assert result.voice_gate is None


class TestForced:
    def test_force_keeps_the_audio_and_records_it_in_the_manifest(
        self, source: Path, profile: VoiceProfile, tmp_path: Path
    ) -> None:
        """--force 的产物必须自带标记。哪次对外内容是绕过闸门发出去的，
        半年后只有 manifest 答得出来。"""
        output = tmp_path / "out" / "示例.wav"

        result = render(
            source,
            profile,
            FakeSynthesizer(),
            RecordingTranscriber("第一段内容。"),
            output=output,
            forced=True,
        )

        assert result.audio_path == output
        assert output.is_file()
        assert result.manifest is not None
        assert result.manifest["forced"] is True
        assert result.manifest["gate_content"]["passed"] is False  # type: ignore[index]

    def test_force_still_writes_the_diff_report(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """绕过闸门不等于放弃证据。"""
        result = render(
            source,
            profile,
            FakeSynthesizer(),
            RecordingTranscriber("第一段内容。"),
            forced=True,
        )

        assert result.diff_path is not None
        assert result.diff_path.is_file()

    def test_force_still_measures_the_voice_gate(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        result = render(
            source,
            profile,
            FakeSynthesizer(),
            RecordingTranscriber("第一段内容。"),
            forced=True,
        )

        assert result.voice_gate is not None


class TestVoiceGateFailure:
    def test_audio_is_still_written_because_only_a_human_can_judge_timbre(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """设计 §9：闸门 2 不过 = 告警 + manifest 标记，仍出音频。

        与闸门 1 不对称是有意的：内容丢失是客观的，音色漂移的判据是代理指标
        （p10 F0），可能误报。人必须能听到那段音频才能判断。
        """
        result = render(
            source,
            profile,
            FakeSynthesizer(freq=250.0),
            perfect_ear(),
        )

        assert result.voice_gate is not None
        assert not result.voice_gate.passed
        assert result.audio_path is not None
        assert result.audio_path.is_file()
        assert result.manifest is not None
        assert result.manifest["gate_voice"]["passed"] is False  # type: ignore[index]


class TestRefusals:
    def test_missing_source_is_refused(self, tmp_path: Path, profile: VoiceProfile) -> None:
        with pytest.raises(FileNotFoundError, match="缺席.md"):
            render(
                tmp_path / "缺席.md",
                profile,
                FakeSynthesizer(),
                perfect_ear(),
            )

    def test_empty_source_is_refused(self, tmp_path: Path, profile: VoiceProfile) -> None:
        path = tmp_path / "空.md"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="没有可朗读"):
            render(path, profile, FakeSynthesizer(), perfect_ear())

    def test_source_that_cleans_down_to_nothing_is_refused(
        self, tmp_path: Path, profile: VoiceProfile
    ) -> None:
        """全是代码块的帖子清洗后无内容。产出一个静音文件是最坏的结果：
        它有正确的文件名、正确的时长，听起来才发现是空的。"""
        path = tmp_path / "只有代码.md"
        path.write_text("```python\nprint(1)\n```\n", encoding="utf-8")

        with pytest.raises(ValueError, match="没有可朗读"):
            render(path, profile, FakeSynthesizer(), perfect_ear())

    def test_a_synthesizer_at_the_wrong_sample_rate_is_refused(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """44.1kHz 被当成 48kHz 解读会让 F0 系统性偏高 8.8%，刚好吃掉闸门 2 的
        6% 余量，表现为「音色漂移」而真实原因是单位错。"""
        with pytest.raises(ValueError, match="44100"):
            render(
                source,
                profile,
                FakeSynthesizer(sample_rate=44100),
                perfect_ear(),
            )


class TestDefaultOutputPath:
    def test_default_path_derives_from_the_source_name(
        self, source: Path, profile: VoiceProfile
    ) -> None:
        """默认落在 out/ 下，文件名沿用帖子名——中文保留，空白折成连字符，
        免得每次引用都要加引号。"""
        result = render(source, profile, FakeSynthesizer(), perfect_ear())

        assert result.audio_path == Path("out/示例-帖子.wav")
        assert result.audio_path.is_file()


def test_result_carries_the_audio_that_was_written(
    source: Path, profile: VoiceProfile
) -> None:
    """闸门 2 用的是内存里的各段，落盘的是拼接结果。两者必须是同一份音频——
    否则闸门在给一段没人会听到的音频打分。"""
    result = render(source, profile, FakeSynthesizer(), perfect_ear())

    assert result.audio_path is not None
    on_disk = wavio.read(result.audio_path)
    assert result.manifest is not None
    assert len(on_disk) == pytest.approx(
        result.manifest["audio_seconds"] * OUTPUT_SAMPLE_RATE, abs=1  # type: ignore[operator]
    )
    assert np.abs(on_disk).max() > 0.1
