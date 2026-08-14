"""命令行入口：参数解析、退出码、人看得懂的失败消息。

退出码是这一层最重要的契约——它决定了这套东西能不能进脚本。三档：

  0  成功（音色闸门可能只是告警）
  1  内容闸门不过：**预期内**的失败，重跑或 --force 可解
  2  操作性错误：档案缺失、输入为空、权重没缓存。argparse 的用法错误也是 2

失败消息必须带可复制的修复命令。一个只说「音色 house 不存在」的错误，
会让人去翻文档；带上铸造命令的错误，人直接粘贴就走。
"""

from pathlib import Path

import pytest
from fakes import P10_OF_200HZ, FakeSynthesizer, RecordingTranscriber, sine

from voxout import cli, markdown, normalize, wavio
from voxout.config import OUTPUT_SAMPLE_RATE
from voxout.profile import mint_profile

SPOKEN = "第一段内容。第二段内容。"
ANCHOR_TEXT = "这是一段用来测试音色是否稳定的中文句子。"


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def voice_root(tmp_path: Path) -> Path:
    """一个真实铸造出来的音色档案——用替身合成器，但走 mint_profile 的全部校验。"""
    root = tmp_path / "voice"
    mint_profile("house", ANCHOR_TEXT, FakeSynthesizer(), seed=42, root=root)
    return root


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "帖子.md"
    path.write_text(SPOKEN, encoding="utf-8")
    return path


def perfect_ear(md: str = SPOKEN) -> RecordingTranscriber:
    return RecordingTranscriber(normalize.to_speakable_numbers(markdown.to_speakable(md)))


def run(argv: list[str], *, synthesizer: object = None, ear: object = None) -> int:
    """跑一次 CLI，把两个昂贵资源换成替身。

    工厂而非实例：CLI 必须在**加载模型之前**就完成参数校验与档案校验，
    否则一个拼错的音色名要等 2 秒模型加载才报错。传工厂才能测出这个顺序。
    """
    return cli.main(
        argv,
        synthesizer_factory=lambda: synthesizer or FakeSynthesizer(),
        transcriber_factory=lambda: ear or perfect_ear(),
    )


class TestRenderCommand:
    def test_successful_render_exits_zero_and_writes_both_files(
        self, source: Path, voice_root: Path, tmp_path: Path, capsys
    ) -> None:
        output = tmp_path / "out" / "成品.wav"

        code = run(
            ["render", str(source), "--voice", "house", "-o", str(output),
             "--voice-root", str(voice_root)]
        )

        assert code == 0
        assert output.is_file()
        assert (tmp_path / "out" / "成品.manifest.json").is_file()
        assert "成品.wav" in capsys.readouterr().out

    def test_report_shows_both_gate_readings_with_their_thresholds(
        self, source: Path, voice_root: Path, capsys
    ) -> None:
        """人要在一屏之内看到「过了多少、阈值多少」。只说「通过」等于没说。"""
        run(["render", str(source), "--voice", "house", "--voice-root", str(voice_root)])

        out = capsys.readouterr().out
        assert "内容" in out
        assert "音色" in out
        assert "97" in out
        assert "6" in out

    def test_content_gate_failure_exits_one_with_the_force_hint(
        self, source: Path, voice_root: Path, capsys
    ) -> None:
        """退出码 1 是「预期内的失败」。消息里要有能直接粘贴的下一步。"""
        code = run(
            ["render", str(source), "--voice", "house", "--voice-root", str(voice_root)],
            ear=RecordingTranscriber("第一段内容。"),
        )

        captured = capsys.readouterr()
        assert code == 1
        assert "--force" in captured.out + captured.err

    def test_voice_gate_failure_warns_but_still_exits_zero(
        self, source: Path, voice_root: Path, capsys
    ) -> None:
        """设计 §9：音色闸门只告警。它的判据是代理指标，可能误报，
        而唯一能下最终判断的是人的耳朵——所以音频必须出，流程不该被挡死。"""
        code = run(
            ["render", str(source), "--voice", "house", "--voice-root", str(voice_root)],
            synthesizer=FakeSynthesizer(freq=250.0),
        )

        captured = capsys.readouterr()
        assert code == 0
        assert "试听" in captured.out + captured.err

    def test_force_turns_a_content_failure_into_a_warning(
        self, source: Path, voice_root: Path
    ) -> None:
        code = run(
            ["render", str(source), "--voice", "house", "--voice-root", str(voice_root),
             "--force"],
            ear=RecordingTranscriber("第一段内容。"),
        )

        assert code == 0
        assert Path("out/帖子.wav").is_file()

    def test_missing_voice_exits_two_with_the_mint_command(
        self, source: Path, tmp_path: Path, capsys
    ) -> None:
        code = run(
            ["render", str(source), "--voice", "缺席", "--voice-root", str(tmp_path / "voice")]
        )

        assert code == 2
        assert "mint-voice 缺席" in capsys.readouterr().err

    def test_missing_source_exits_two(self, voice_root: Path, capsys) -> None:
        code = run(
            ["render", "缺席.md", "--voice", "house", "--voice-root", str(voice_root)]
        )

        assert code == 2
        assert "缺席.md" in capsys.readouterr().err

    def test_empty_source_exits_two_rather_than_producing_silence(
        self, tmp_path: Path, voice_root: Path, capsys
    ) -> None:
        path = tmp_path / "空.md"
        path.write_text("```python\nprint(1)\n```\n", encoding="utf-8")

        code = run(["render", str(path), "--voice", "house", "--voice-root", str(voice_root)])

        assert code == 2
        assert "可朗读" in capsys.readouterr().err

    def test_the_profile_is_validated_before_the_model_is_loaded(
        self, source: Path, tmp_path: Path
    ) -> None:
        """拼错音色名不该等 2 秒模型加载才报错。"""
        loaded: list[str] = []

        def exploding_factory() -> object:
            loaded.append("模型被加载了")
            return FakeSynthesizer()

        code = cli.main(
            ["render", str(source), "--voice", "缺席", "--voice-root", str(tmp_path)],
            synthesizer_factory=exploding_factory,
            transcriber_factory=perfect_ear,
        )

        assert code == 2
        assert loaded == []


class TestMintVoiceCommand:
    def test_minting_creates_a_complete_profile(self, tmp_path: Path, capsys) -> None:
        root = tmp_path / "voice"

        code = run(["mint-voice", "house", "--seed", "42", "--voice-root", str(root)])

        assert code == 0
        assert (root / "house" / "anchor.wav").is_file()
        assert (root / "house" / "anchor.txt").is_file()
        assert (root / "house" / "profile.toml").is_file()
        assert "house" in capsys.readouterr().out

    def test_the_default_anchor_text_is_the_calibrated_one(self, tmp_path: Path) -> None:
        """设计 §4 的探测就是用这句话做的，它的 bit 级可复现性已实测。
        换一句就要重新标定，所以默认值必须是那一句。"""
        root = tmp_path / "voice"

        run(["mint-voice", "house", "--voice-root", str(root)])

        assert (root / "house" / "anchor.txt").read_text(encoding="utf-8") == ANCHOR_TEXT

    def test_custom_anchor_text_is_honored(self, tmp_path: Path) -> None:
        root = tmp_path / "voice"

        run(["mint-voice", "house", "--text", "另一句锚点文本。", "--voice-root", str(root)])

        assert (root / "house" / "anchor.txt").read_text(encoding="utf-8") == "另一句锚点文本。"

    def test_the_baseline_is_recorded_so_gate_two_has_a_denominator(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "voice"

        run(["mint-voice", "house", "--voice-root", str(root)])

        toml = (root / "house" / "profile.toml").read_text(encoding="utf-8")
        assert "baseline_p10_f0" in toml
        assert str(round(P10_OF_200HZ, 3))[:6] in toml

    def test_refusing_to_overwrite_exits_two(self, voice_root: Path, capsys) -> None:
        """覆盖 house 会静默改变此后每一次对外输出的声音。"""
        code = run(["mint-voice", "house", "--voice-root", str(voice_root)])

        assert code == 2
        assert "已存在" in capsys.readouterr().err


class TestUsage:
    def test_no_subcommand_exits_two(self, capsys) -> None:
        assert run([]) == 2

    def test_unknown_subcommand_exits_two(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            run(["转录", "x.md"])

        assert exit_info.value.code == 2


def test_anchor_geometry_is_enforced_on_load(tmp_path: Path, capsys) -> None:
    """44.1kHz 的锚点会让 F0 系统性偏高 8.8%，刚好吃掉闸门 2 的 6% 余量。
    症状是「音色漂移」，真实原因是单位错——所以加载时就要挡住，并给出重采样命令。"""
    root = tmp_path / "voice"
    mint_profile("house", ANCHOR_TEXT, FakeSynthesizer(), root=root)
    wavio.write(root / "house" / "anchor.wav", sine(200.0, 1.0, 44100), 44100)
    source = tmp_path / "帖子.md"
    source.write_text(SPOKEN, encoding="utf-8")

    code = run(["render", str(source), "--voice", "house", "--voice-root", str(root)])

    assert code == 2
    error = capsys.readouterr().err
    assert "44100" in error
    assert "ffmpeg" in error
    assert str(OUTPUT_SAMPLE_RATE) in error
