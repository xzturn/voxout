"""音色档案。

设计 §4 的结论是「锚点是正确性要求，不是可选项」：零样本模式下同一段文本
两次调用的 p10 F0 极差达 53.9%，等于每次换一个人说话。档案就是把那个人
固定下来的载体，所以它的校验必须严——一个坏档案会污染此后所有对外内容。

这里全部用合成正弦当锚点，不碰模型。
"""

import wave
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

from voxout import wavio
from voxout.profile import VoiceProfile, load_profile, mint_profile

ANCHOR_TEXT = "这是一段用来测试音色是否稳定的中文句子。"


def sine(freq: float = 180.0, seconds: float = 2.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@dataclass(frozen=True)
class FakeSegment:
    audio: np.ndarray
    sample_rate: int


class FakeSynthesizer:
    """替身：只需满足 mint_profile 用到的那一个方法。"""

    def __init__(self, audio: np.ndarray, sample_rate: int = 48000) -> None:
        self._audio = audio
        self._sample_rate = sample_rate
        self.calls: list[tuple[str, int | None]] = []

    def generate(self, text: str, *, seed: int | None = None) -> FakeSegment:
        self.calls.append((text, seed))
        return FakeSegment(audio=self._audio, sample_rate=self._sample_rate)


def make_profile_dir(
    root: Path,
    name: str = "house",
    *,
    anchor: np.ndarray | None = None,
    sample_rate: int = 48000,
    anchor_text: str = ANCHOR_TEXT,
    baseline: float | None = 180.0,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    wavio.write(directory / "anchor.wav", sine() if anchor is None else anchor, sample_rate)
    (directory / "anchor.txt").write_text(anchor_text, encoding="utf-8")

    lines = [f'name = "{name}"', 'source = "minted"', 'created = "2026-08-14T10:00:00+00:00"']
    if baseline is not None:
        lines.append(f"baseline_p10_f0 = {baseline}")
    (directory / "profile.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


class TestLoadValidProfile:
    def test_all_fields_are_populated(self, tmp_path: Path) -> None:
        make_profile_dir(tmp_path)
        loaded = load_profile("house", root=tmp_path)

        assert loaded.name == "house"
        assert loaded.anchor_text == ANCHOR_TEXT
        assert loaded.baseline_p10_f0 == 180.0
        assert loaded.anchor_wav == tmp_path / "house" / "anchor.wav"
        assert loaded.source == "minted"

    def test_profile_is_immutable(self, tmp_path: Path) -> None:
        make_profile_dir(tmp_path)
        loaded = load_profile("house", root=tmp_path)

        with pytest.raises(FrozenInstanceError):
            loaded.baseline_p10_f0 = 200.0  # type: ignore[misc]


class TestValidationFailures:
    def test_missing_voice_directory_names_the_mint_command(self, tmp_path: Path) -> None:
        """错误消息必须含可直接复制的修复命令（设计 §9）。"""
        with pytest.raises(FileNotFoundError, match="mint-voice"):
            load_profile("house", root=tmp_path)

    def test_missing_anchor_wav_names_the_mint_command(self, tmp_path: Path) -> None:
        make_profile_dir(tmp_path)
        (tmp_path / "house" / "anchor.wav").unlink()

        with pytest.raises(FileNotFoundError, match="mint-voice"):
            load_profile("house", root=tmp_path)

    def test_missing_anchor_text_names_the_mint_command(self, tmp_path: Path) -> None:
        make_profile_dir(tmp_path)
        (tmp_path / "house" / "anchor.txt").unlink()

        with pytest.raises(FileNotFoundError, match="mint-voice"):
            load_profile("house", root=tmp_path)

    def test_wrong_sample_rate_names_the_resample_command(self, tmp_path: Path) -> None:
        """模型内部会把参考音降到 16kHz，它不在乎采样率；**我们的 LTAS 比较在乎**。

        44.1kHz 的锚点与 48kHz 的输出做频谱比较时带宽不同，会表现为莫名的音色漂移。
        """
        make_profile_dir(tmp_path, sample_rate=44100)

        with pytest.raises(ValueError, match="ffmpeg") as excinfo:
            load_profile("house", root=tmp_path)
        assert "48000" in str(excinfo.value)

    def test_stereo_anchor_names_the_downmix_command(self, tmp_path: Path) -> None:
        directory = tmp_path / "house"
        directory.mkdir(parents=True)
        with wave.open(str(directory / "anchor.wav"), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(np.zeros(2000, dtype=np.int16).tobytes())
        (directory / "anchor.txt").write_text(ANCHOR_TEXT, encoding="utf-8")
        (directory / "profile.toml").write_text("baseline_p10_f0 = 180.0\n", encoding="utf-8")

        with pytest.raises(ValueError, match="ffmpeg"):
            load_profile("house", root=tmp_path)

    def test_empty_anchor_text_is_refused(self, tmp_path: Path) -> None:
        """ref_text 为空会退化成非克隆模式，音色随机——这是静默的正确性事故。"""
        make_profile_dir(tmp_path, anchor_text="   \n")

        with pytest.raises(ValueError, match="anchor.txt"):
            load_profile("house", root=tmp_path)

    def test_missing_baseline_is_refused(self, tmp_path: Path) -> None:
        make_profile_dir(tmp_path, baseline=None)

        with pytest.raises(ValueError, match="baseline_p10_f0"):
            load_profile("house", root=tmp_path)

    def test_non_positive_baseline_is_refused(self, tmp_path: Path) -> None:
        """基线是漂移百分比的分母，为 0 会让闸门 2 产出 inf 或 nan。"""
        make_profile_dir(tmp_path, baseline=0.0)

        with pytest.raises(ValueError, match="baseline_p10_f0"):
            load_profile("house", root=tmp_path)


class TestMint:
    def test_minted_profile_loads_back_identically(self, tmp_path: Path) -> None:
        synthesizer = FakeSynthesizer(sine(180.0))

        minted = mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)
        loaded = load_profile("house", root=tmp_path)

        assert minted == loaded
        assert loaded.seed == 42
        assert loaded.mint_text == ANCHOR_TEXT
        assert loaded.source == "minted"

    def test_baseline_is_measured_from_the_minted_audio(self, tmp_path: Path) -> None:
        minted = mint_profile(
            "house", ANCHOR_TEXT, FakeSynthesizer(sine(180.0)), seed=42, root=tmp_path
        )
        assert minted.baseline_p10_f0 == pytest.approx(180.0, rel=0.01)

    def test_seed_is_passed_through_for_reproducibility(self, tmp_path: Path) -> None:
        """锚点的 bit 级可复现性依赖固定种子（设计 §3）。"""
        synthesizer = FakeSynthesizer(sine())
        mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        assert synthesizer.calls == [(ANCHOR_TEXT, 42)]

    def test_existing_profile_is_never_silently_overwritten(self, tmp_path: Path) -> None:
        """覆盖 house 音色会静默改变此后每一次渲染的声音。必须显式删除。"""
        make_profile_dir(tmp_path)

        with pytest.raises(FileExistsError, match="house"):
            mint_profile("house", ANCHOR_TEXT, FakeSynthesizer(sine()), seed=42, root=tmp_path)

    def test_silent_audio_cannot_become_an_anchor(self, tmp_path: Path) -> None:
        """全静音测不出 p10 F0（nan）。写成档案后闸门 2 会永久比对 nan。"""
        silent = FakeSynthesizer(np.zeros(48000, dtype=np.float32))

        with pytest.raises(ValueError, match="浊音"):
            mint_profile("house", ANCHOR_TEXT, silent, seed=42, root=tmp_path)

    def test_wrong_sample_rate_from_synthesizer_is_refused(self, tmp_path: Path) -> None:
        odd = FakeSynthesizer(sine(180.0, sr=24000), sample_rate=24000)

        with pytest.raises(ValueError, match="48000"):
            mint_profile("house", ANCHOR_TEXT, odd, seed=42, root=tmp_path)

    def test_empty_mint_text_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            mint_profile("house", "  ", FakeSynthesizer(sine()), seed=42, root=tmp_path)

    def test_validation_failure_never_creates_the_directory(self, tmp_path: Path) -> None:
        """校验全部发生在建目录之前，失败时磁盘上什么都不留。"""
        silent = FakeSynthesizer(np.zeros(48000, dtype=np.float32))

        with pytest.raises(ValueError):
            mint_profile("house", ANCHOR_TEXT, silent, seed=42, root=tmp_path)

        assert not (tmp_path / "house").exists()

    def test_write_failure_midway_cleans_up_the_partial_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """半个档案比没有档案更糟：load 会通过一部分校验然后在别处炸。

        模拟写盘中途失败（磁盘满、权限），确认目录被回收而不是留下残骸。
        """
        import voxout.profile as profile_module

        def fail_after_wav(*_args: object, **_kwargs: object) -> None:
            raise OSError("模拟磁盘写入失败")

        monkeypatch.setattr(profile_module.wavio, "write", fail_after_wav)

        with pytest.raises(OSError):
            mint_profile("house", ANCHOR_TEXT, FakeSynthesizer(sine()), seed=42, root=tmp_path)

        assert not (tmp_path / "house").exists()


def test_voice_profile_can_be_constructed_directly() -> None:
    """数据类本身不做校验——校验属于 load/mint 的边界职责。"""
    profile = VoiceProfile(
        name="x",
        anchor_wav=Path("/tmp/a.wav"),
        anchor_text="文本",
        baseline_p10_f0=180.0,
        source="recorded",
        seed=None,
        mint_text=None,
        created="2026-08-14T10:00:00+00:00",
    )
    assert profile.seed is None
