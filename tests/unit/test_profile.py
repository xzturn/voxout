"""音色档案。

设计 §4 的结论是「锚点是正确性要求，不是可选项」：零样本模式下同一段文本
两次调用的 p10 F0 极差达 53.9%，等于每次换一个人说话。档案就是把那个人
固定下来的载体，所以它的校验必须严——一个坏档案会污染此后所有对外内容。

这里全部用合成正弦当锚点，不碰模型。
"""

import logging
import wave
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import numpy as np
import pytest

from voxout import config, wavio
from voxout.metrics import p10_f0
from voxout.profile import VoiceProfile, load_profile, mint_profile

ANCHOR_TEXT = "这是一段用来测试音色是否稳定的中文句子。"


def sine(freq: float = 180.0, seconds: float = 2.0, sr: int = 48000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@dataclass(frozen=True)
class FakeSegment:
    audio: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class MintCall:
    """一次 generate 调用里，铸造流程能被观察到的全部行为。"""

    text: str
    seed: int | None
    ref_text: str | None
    ref_audio_on_disk: bool


class FakeSynthesizer:
    """替身：按调用顺序吐出预置音频，并记下每次调用看到的参考音状态。

    音高逐次不同是必要的：铸造现在生成 1 段锚点 + N 段条件采样，只有每次返回
    不同音高，才验得出基线取的是中位数而不是第一个或平均值。给不够时重复用
    最后一段。

    `ref_audio_on_disk` 记的是**调用那一刻**参考音是否已在盘上。条件采样必须
    读得到锚点，而锚点是同一次铸造里刚写的——顺序错了模型会静默退化成零样本，
    基线于是变成随机说话人的读数（实测极差 48.82%）。
    """

    def __init__(self, *audios: np.ndarray, sample_rate: int = 48000) -> None:
        if not audios:
            raise ValueError("替身至少要给一段音频")
        self._audios = audios
        self._sample_rate = sample_rate
        self.calls: list[MintCall] = []

    def generate(
        self,
        text: str,
        *,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        seed: int | None = None,
    ) -> FakeSegment:
        self.calls.append(
            MintCall(
                text=text,
                seed=seed,
                ref_text=ref_text,
                ref_audio_on_disk=ref_audio is not None and ref_audio.is_file(),
            )
        )
        audio = self._audios[min(len(self.calls) - 1, len(self._audios) - 1)]
        return FakeSegment(audio=audio, sample_rate=self._sample_rate)


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

    def test_missing_profile_toml_names_the_mint_command(self, tmp_path: Path) -> None:
        """没有元数据就没有基线，闸门 2 失去参照——不能退化成「只看段间极差」。"""
        (make_profile_dir(tmp_path) / "profile.toml").unlink()

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

    def test_seed_is_passed_through_for_reproducibility(self, tmp_path: Path) -> None:
        """锚点的 bit 级可复现性依赖固定种子（设计 §3）。"""
        synthesizer = FakeSynthesizer(sine())
        mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        assert synthesizer.calls[0] == MintCall(
            text=ANCHOR_TEXT, seed=42, ref_text=None, ref_audio_on_disk=False
        )


class TestBaselineIsMeasuredOnConditionedGenerations:
    """基线要与闸门 2 实际测量的东西同分布，且不能是单次读数。

    两件事都做了，但证据强度不同，这里按实测写清楚，别让弱的那条搭强的那条的车：

    **同分布**是原则，不是修 bug。旧定义量锚点音频自身，渲染时量的是锚点条件下的
    生成，拿一个当另一个的分母是跨分布比较。但实测两者只差 0.31%（166.32 →
    165.81），所以它解释不了真实内容里 23 段有 22 段漂移为负、中位数 −5.41%。

    **n=1** 才是那个缺陷。文本与说话人固定、只换采样，p10 F0 极差实测 7.71% 与
    14.91%；单次读数落在约 ±15% 宽的分布里，n=1 的基线把整个宽度原样传给此后
    每一次判定。−5.41% 也落在这个宽度之内——它是噪声加真实文本与锚点文本本身的
    差别，不是分布错配的签名。故取 N 份的中位数。
    """

    def test_baseline_is_the_median_of_the_conditioned_readings(self, tmp_path: Path) -> None:
        """250 / 200 / 190 → 中位数 200。均值是 213，第一份是 250，都被这条排除。"""
        synthesizer = FakeSynthesizer(sine(180.0), sine(250.0), sine(200.0), sine(190.0))

        minted = mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        assert minted.baseline_p10_f0 == pytest.approx(200.0, rel=0.01)

    def test_the_anchor_audio_itself_is_not_the_baseline(self, tmp_path: Path) -> None:
        """锚点自身 300 Hz、条件采样全 200 Hz——基线必须是 200。"""
        synthesizer = FakeSynthesizer(sine(300.0), sine(200.0))

        minted = mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        assert minted.baseline_p10_f0 == pytest.approx(200.0, rel=0.01)

    def test_conditioned_samples_are_generated_against_the_anchor_on_disk(
        self, tmp_path: Path
    ) -> None:
        """条件采样若拿不到锚点就退化成零样本，基线成了随机说话人的读数。"""
        synthesizer = FakeSynthesizer(sine(180.0), sine(200.0))

        mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        conditioned = synthesizer.calls[1:]
        assert len(conditioned) == config.BASELINE_SAMPLES
        for call in conditioned:
            assert call.ref_audio_on_disk, "锚点必须先落盘，条件采样才读得到"
            assert call.ref_text == ANCHOR_TEXT
            assert call.text == ANCHOR_TEXT

    def test_conditioned_samples_use_distinct_seeds(self, tmp_path: Path) -> None:
        """共用一个种子会让 N 份采样完全相同，中位数悄悄退化成 n=1——一个「看起来
        是 N 个采样」的 1 个采样，正是这次标定要修的那类缺陷。"""
        synthesizer = FakeSynthesizer(sine(180.0), sine(200.0))

        mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        seeds = [call.seed for call in synthesizer.calls]
        assert len(set(seeds)) == len(seeds), f"种子有重复：{seeds}"

    def test_the_spread_of_the_readings_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """各份读数的极差就是闸门 2 的噪声下限，必须留在铸造记录里。

        只写中位数等于把一个不确定的数说成确定的：同一份中位数，极差 1% 与极差
        30% 对阈值的含义完全不同，而铸造之后没人能再重建这三份读数。
        """
        conditioned = (sine(190.0), sine(200.0), sine(210.0))
        synthesizer = FakeSynthesizer(sine(180.0), *conditioned)

        with caplog.at_level(logging.INFO, logger="voxout.profile"):
            mint_profile("house", ANCHOR_TEXT, synthesizer, seed=42, root=tmp_path)

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "极差" in logged, f"铸造没有记录读数离散度：{logged!r}"
        # 期望值现算而不写死：190 Hz 正弦的 p10 读数不必恰好是 190.0，那点误差
        # 属于估计器精度，不该让一条「读数有没有留痕」的断言跟着它一起碎。
        for audio in conditioned:
            reading = f"{p10_f0(audio, config.OUTPUT_SAMPLE_RATE):.1f}"
            assert reading in logged, f"读数 {reading} 没有留痕：{logged!r}"

    def test_seedless_minting_leaves_every_sample_unseeded(self, tmp_path: Path) -> None:
        """不给种子就一份都不钉——从 None 派生出数字会假装可复现。"""
        synthesizer = FakeSynthesizer(sine(180.0), sine(200.0))

        mint_profile("house", ANCHOR_TEXT, synthesizer, root=tmp_path)

        assert [call.seed for call in synthesizer.calls] == [None] * (
            1 + config.BASELINE_SAMPLES
        )

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

        必须放过**第一次** `wavio.write`：那一次写的是临时目录里的暂存锚点
        （`mint_profile` 用它当参考音去测基线），发生在正式目录建立之前。
        一律失败的替身会停在那里，正式目录根本没建过——断言「目录不存在」照样
        通过，但清理路径一行都没跑到。测试于是能在清理逻辑被删掉之后依然全绿。

        末态无法区分「清理跑了」和「压根没建」，所以这里在**失败当刻**观察一次
        目录是否存在，用中间态把两者分开。
        """
        import voxout.profile as profile_module

        real_write = profile_module.wavio.write
        directory = tmp_path / "house"
        observed: list[bool] = []

        def fail_on_the_profile_write(path: Path, *args: object, **kwargs: object) -> None:
            if not observed:  # 第一次 = 临时目录里的暂存锚点，放过
                observed.append(directory.exists())
                real_write(path, *args, **kwargs)
                return
            observed.append(directory.exists())
            raise OSError("模拟磁盘写入失败")

        monkeypatch.setattr(profile_module.wavio, "write", fail_on_the_profile_write)

        with pytest.raises(OSError):
            mint_profile("house", ANCHOR_TEXT, FakeSynthesizer(sine()), seed=42, root=tmp_path)

        assert observed == [False, True], "第二次写必须发生在正式目录建立之后"
        assert not directory.exists()


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
