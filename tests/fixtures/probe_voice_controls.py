"""探测：闸门 2 的正/负控制标尺，以及判据该取中位数还是最大值。

闸门 2 的阈值只有放在标尺上才有意义。标尺的两端必须由控制条件定义，而不是由
印象定义：

    地板    **挂锚点**的一次连续生成，切成若干段。音色由构造保证相同——一次
            发声、一条采样轨迹。读数即「同一个人念不同内容」的不可约抖动。
    天花板  零样本各生成一次。实测每次换一个人，这是「还能算同一个人」的上界。
    工况    挂锚点各生成一次。真实渲染就走这条。

地板也挂锚点，是为了让它与工况**只差一个变量**：一次调用对 N 次调用。闸门 2 判的
恰好是后者会不会把音色打散；地板若不挂锚点，两者之间就混进「有没有锚点」这第二个
变量，差多少都说不清是哪一个造成的。

三个条件里只有挂了锚点的两个能报「对基线漂移」。零样本每次抽一个随机说话人，它离
我们的基线多远纯属抽签——实测抽到 160.2 / 106.4 / 128.7 Hz，160.2 几乎正中 165.81
的基线，中位漂移于是只读出 22.41%，低于同一批音频自己的段间极差 40.85%。若三段都
抽到 165 附近，漂移会读出接近 0，而「换了个人」这件事在 p10 上就隐形了。同一批音频
的 LTAS 余弦是 0.586，毫不含糊——这是设计 §4 坚持用两个独立判别器的直接证据，本轮
第一次真正撞上。

地板那一次长生成可能跳读（T13a：单次调用超过 60 汉字就开始丢内容）。**不影响结论**：
它量的是身份不是内容，跳掉一句之后剩下的仍是同一次发声。若哪天要用这个脚本量内容，
才需要改成逐段送。

首轮读数（house，seed 42，samples=3）：

    地板   p10 极差  7.33%   median|漂移| 3.82%   LTAScos min 0.919
    工况   p10 极差 11.39%   median|漂移| 4.89%   LTAScos min 0.870
    天花板 p10 极差 40.85%   （漂移不适用）      LTAScos min 0.586

三点结论：

**锚点补掉了 87.9% 的缺口。** 极差从换人的 40.85% 压到 11.39%，而地板是 7.33%，
即 (40.85−11.39)/(40.85−7.33)。设计 §4 用另一套统计量（对固定文本的极差）给
`ref_audio` 记的位置是 87.7%——两次独立测量、不同语料、不同统计量，落在同一处。

**闸门 2 在真实内容上读到的数，大部分是地板本身。** 工况中位漂移 4.89%，地板
3.82%，多次调用只额外贡献约 1.1 个百分点；两篇真实短帖读 4.72–5.35%，与工况一致。
15% 的阈值因此是在一个**无法再压下去的**地板之上留约 3 倍余量——把阈值往 4% 那边
收紧不会换来更强的检测，只会开始误报地板。

**换人时 LTAS 比 p10 硬。** 0.919 → 0.870 → 0.586，而 p10 漂移在换人那一列根本
不能用（抽签）。闸门 2 目前只判 p10，这是它的已知弱点，LTAS 升为判据是待办
（`docs/listening-review.md`）。

原始版本（`tests/probe_control.py`，已删除）自带一份 F0 与极差实现，且负控制读的是
`/tmp` 下上一次探测留下的 wav——那些文件早已不在，脚本其实跑不动了。现在三个条件
全部在一次运行里现生成，度量一律从 `voxout.metrics` 导入：探测与生产必须共用同一
把尺子，否则探测标定出来的阈值管不到生产。

用法：
    uv run python tests/fixtures/probe_voice_controls.py [--voice house] [--samples 3]

`--samples` 加大可复现「极值统计量随段数长、中位数不随」这件事（真实语料上的读数
见 `verify.check_voice`）。闸门 2 因此判中位数。
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from voxout import metrics
from voxout.config import OUTPUT_SAMPLE_RATE, VOICE_DRIFT_MAX_PCT
from voxout.profile import VoiceProfile, load_profile
from voxout.synth import Synthesizer

TEXTS = (
    "第一段说的是选型的结论和它的代价。",
    "第二段解释为什么机器性能不是真正的瓶颈。",
    "第三段留给尚未验证的风险和后续计划。",
)
"""与设计 §4 那批探测同一组文本。换文本就与已记录的 0.9% / 53.9% 不可比了。"""

VOICE_DEFAULT = "house"
SAMPLES_DEFAULT = 3
SEED_DEFAULT = 42


@dataclass(frozen=True)
class Reading:
    """一个控制条件下的全部读数。"""

    label: str
    p10s: tuple[float, ...]
    spread_pct: float
    cos_min: float
    cos_mean: float
    drift_is_meaningful: bool
    """「对基线漂移」这一列在本条件下是否成立。

    只有**以本音色的锚点为条件**生成时才成立。负控制每次抽一个随机说话人，它离
    我们的基线多远纯属抽签：实测三段抽到 160.2 / 106.4 / 128.7 Hz，其中 160.2
    几乎正中 165.81 的基线，中位漂移于是只读出 22.41%，远低于同一批音频的段间
    极差 40.85%。若三段恰好都抽到 165 附近，漂移会读出接近 0，而 LTAS 余弦照旧
    是 0.586——「换了个人」在这个指标上会隐形。

    把三个条件的漂移并排印出来，等于邀请读者做一次跨条件比较，而这正是闸门 2
    原先那个 6% 阈值的成因（3.4% 测于文本固定的条件，阈值管的是真实内容）。
    所以不成立就不印。
    """


def _text_at(index: int) -> str:
    """第 index 个片段的文本。循环取用，保证各片段内容不同——这些条件量的是
    「同一个人念不同内容」，用同一句话会把内容差异这一项从读数里抹掉。"""
    return TEXTS[index % len(TEXTS)]


def _seed_at(seed: int | None, index: int) -> int | None:
    return None if seed is None else seed + index


def _describe(
    label: str, clips: tuple[np.ndarray, ...], *, drift_is_meaningful: bool
) -> Reading:
    """把一组音频折成一行读数：p10 极差 + LTAS 余弦。

    两个判别器互相独立：p10 是音高地板（身份的粗粒度指标），LTAS 余弦是频谱
    指纹（音色）。一个指标抽风时另一个还在，这是设计 §4 定下的做法——本轮实测
    直接证明了这条：负控制的 p10 中位漂移只有 22.41%（抽签抽到一个近基线的
    说话人），而 LTAS 余弦 0.586 毫不含糊。
    """
    p10s = tuple(metrics.p10_f0(clip, OUTPUT_SAMPLE_RATE) for clip in clips)
    spectra = [metrics.ltas(clip, OUTPUT_SAMPLE_RATE) for clip in clips]
    cosines = [
        metrics.cosine(spectra[i], spectra[j])
        for i in range(len(spectra))
        for j in range(i + 1, len(spectra))
    ]
    return Reading(
        label=label,
        p10s=p10s,
        spread_pct=metrics.spread_pct(list(p10s)),
        cos_min=float(np.min(cosines)) if cosines else float("nan"),
        cos_mean=float(np.mean(cosines)) if cosines else float("nan"),
        drift_is_meaningful=drift_is_meaningful,
    )


def _abs_drifts(p10s: tuple[float, ...], baseline: float) -> np.ndarray:
    return np.abs(np.asarray(p10s) - baseline) / baseline * 100.0


def _print_reading(reading: Reading, baseline: float) -> None:
    if reading.drift_is_meaningful:
        drifts = _abs_drifts(reading.p10s, baseline)
        drift_column = (
            f"对基线 median {np.median(drifts):6.2f}%  max {np.max(drifts):6.2f}%"
        )
    else:
        drift_column = "对基线 ——（随机说话人，见 Reading.drift_is_meaningful）"
    print(
        f"  {reading.label:<22}"
        f"p10 极差 {reading.spread_pct:6.2f}%   {drift_column}   "
        f"LTAScos min {reading.cos_min:.3f} mean {reading.cos_mean:.3f}",
        flush=True,
    )
    print(f"    各段 p10  {' '.join(f'{v:.1f}' for v in reading.p10s)}", flush=True)


def _positive_control(
    synthesizer: Synthesizer, samples: int, seed: int | None, profile: VoiceProfile
) -> tuple[np.ndarray, ...]:
    """一次连续生成，切成 samples 段。音色由构造保证相同。

    这一次生成**也挂锚点**，与工况唯一的差别就是「一次调用」对「N 次调用」。
    闸门 2 要判的恰好是后者会不会把音色打散，所以地板必须在同样的条件化下量，
    否则地板与工况之间混进了「有没有锚点」这第二个变量，读数差多少都说明不了
    是哪一个造成的。

    代价：切分按等长样本数，不按句边界，所以每一份里落进多少句末下降调是随机的
    ——地板读数因此包含这一项，它本来也是真实渲染里躲不掉的（p10 是低端分位）。
    """
    joined = "".join(_text_at(i) for i in range(samples))
    segment = synthesizer.generate(
        joined, ref_audio=profile.anchor_wav, ref_text=profile.anchor_text, seed=seed
    )
    audio = np.asarray(segment.audio, dtype=np.float32)
    width = len(audio) // samples
    return tuple(audio[i * width : (i + 1) * width] for i in range(samples))


def _independent_clips(
    synthesizer: Synthesizer,
    samples: int,
    seed: int | None,
    profile: VoiceProfile | None,
) -> tuple[np.ndarray, ...]:
    """各片段一次独立生成。`profile` 为 None 即零样本（负控制）。"""
    return tuple(
        np.asarray(
            synthesizer.generate(
                _text_at(index),
                ref_audio=None if profile is None else profile.anchor_wav,
                ref_text=None if profile is None else profile.anchor_text,
                seed=_seed_at(seed, index),
            ).audio,
            dtype=np.float32,
        )
        for index in range(samples)
    )


def _verdict(anchored: Reading, positive: Reading, negative: Reading, baseline: float) -> int:
    """锚点工况必须落在地板一侧。返回进程退出码。"""
    drifts = _abs_drifts(anchored.p10s, baseline)
    floor_drifts = _abs_drifts(positive.p10s, baseline)
    print(
        f"\n  p10 极差标尺：地板（一次生成）{positive.spread_pct:.2f}%"
        f"  ←  工况（N 次生成）{anchored.spread_pct:.2f}%"
        f"  →  天花板（换人）{negative.spread_pct:.2f}%",
        flush=True,
    )
    print(
        f"  LTAS 标尺  ：{positive.cos_min:.3f}  ←  {anchored.cos_min:.3f}"
        f"  →  {negative.cos_min:.3f}（换人时它比 p10 硬得多）",
        flush=True,
    )
    print(
        f"  闸门 2 判据（两者都挂锚点，可比）：地板 median|漂移| "
        f"{np.median(floor_drifts):.2f}%  →  工况 {np.median(drifts):.2f}%"
        f"，阈值 {VOICE_DRIFT_MAX_PCT:.1f}%",
        flush=True,
    )
    print(
        f"  同一批音频若判 max|漂移| 则读 {np.max(drifts):.2f}%（samples="
        f"{len(drifts)}）。极值随段数长这件事由真实语料量到：6 段 10.65%、"
        f"14 段 13.79%，而中位数是 5.35% 与 4.72%。加大 --samples 可复现。",
        flush=True,
    )

    if not positive.spread_pct < anchored.spread_pct < negative.spread_pct:
        print(
            "\n  ✗ 工况没有落在地板与天花板之间——标尺本身失效，先查这个，"
            "别急着改阈值。",
            flush=True,
        )
        return 1
    if not np.median(drifts) <= VOICE_DRIFT_MAX_PCT:
        print("\n  ✗ 工况自己就过不了闸门 2。锚点或基线出了问题。", flush=True)
        return 1
    print("\n  ✓ 工况落在地板一侧且通过闸门 2，标尺成立。", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=VOICE_DEFAULT, help="音色名")
    parser.add_argument(
        "--samples", type=int, default=SAMPLES_DEFAULT, help="每个条件生成几个片段"
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="基种子，各片段派生")
    args = parser.parse_args(argv)

    if args.samples < 2:
        print("samples 至少 2：一个片段谈不上离散度。", file=sys.stderr)
        return 2

    profile = load_profile(args.voice)
    synthesizer = Synthesizer()
    baseline = profile.baseline_p10_f0
    print(
        f"音色 {profile.name}  基线 p10 {baseline:.2f} Hz  "
        f"每条件 {args.samples} 片段\n",
        flush=True,
    )

    readings = (
        _describe(
            "地板 锚点+一次生成",
            _positive_control(synthesizer, args.samples, args.seed, profile),
            drift_is_meaningful=True,
        ),
        _describe(
            "天花板 零样本换人",
            _independent_clips(synthesizer, args.samples, args.seed, profile=None),
            drift_is_meaningful=False,
        ),
        _describe(
            "工况 锚点+N 次生成",
            _independent_clips(synthesizer, args.samples, args.seed, profile=profile),
            drift_is_meaningful=True,
        ),
    )
    for reading in readings:
        _print_reading(reading, baseline)

    positive, negative, anchored = readings
    return _verdict(anchored, positive, negative, baseline)


if __name__ == "__main__":
    sys.exit(main())
