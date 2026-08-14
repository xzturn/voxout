"""探测：单次生成的文本长度与内容保真度的关系。

起因是一次真实失败。`2026-07-22 留意波动` 一帖 213 汉字、单段送出，三次独立采样
全部在**同一处**跳掉同一段 35 汉字（覆盖率 83.57% / 83.57% / 82.63%）。三次采样
跳同一段，说明这不是采样噪声，而是输入长度或结构本身触发的。

设计文档给闸门 2 之外的 `CHUNK_THRESHOLD_HAN = 900` 的依据是「实测 1050 汉字单次
调用零截断」。那次探测测的是**截断**——音频有没有提前结束；它测不到**跳读**——
音频长度正常、听起来完整，中间少了一句。当时闸门 1（ASR 回环）还不存在，所以
跳读这件事从未被测量过。这个脚本补上那次测量。

用法：
    uv run python tests/fixtures/probe_chunk_fidelity.py [帖子.md] [--thresholds 900,80,60]

它对同一篇帖子扫几个 `threshold_han`，每段单独过一次 ASR 回环，逐段打印覆盖率与
段长。要的不是「过/不过」这一个比特，而是**段长与保真度的关系**——标定一个新阈值
需要知道跳读从多长开始，单点读数给不了这个。

负控制是 900（当前配置，预期复现 ~83%），正控制是最小的那个阈值（预期 ≥97%）。
两端都要出现，读数才可信：只跑一种切法无法区分「切段有效」与「这次运气好」。
"""

import argparse
import sys
from pathlib import Path

from voxout import chunker, markdown, normalize, textmatch, wavio
from voxout.config import OUTPUT_SAMPLE_RATE
from voxout.profile import VoiceProfile, load_profile
from voxout.synth import Synthesizer
from voxout.verify import TranscriberLike, check_content, load_transcriber

POST_DEFAULT = (
    Path.home()
    / "Life OS/60-69 Content Production/63 Short-form Posts"
    / "2026-07-22 留意波动，把控制权拿回来.md"
)
VOICE = "house"
PROBE_DIR = Path("out/probe")

THRESHOLDS_DEFAULT = (900, 200, 80)
"""默认扫描点。900 是当前配置（单段，负控制）；80 约合一个自然段（正控制）。
中间点 200 用来看跳读是渐变还是突变——只有两点无法区分这两种形状。"""


def _read_post(path: Path) -> str:
    return normalize.to_speakable_numbers(markdown.to_speakable(path.read_text(encoding="utf-8")))


def _speak_and_hear(
    label: str,
    chunks: tuple[str, ...],
    synthesizer: Synthesizer,
    transcriber: TranscriberLike,
    profile: VoiceProfile,
) -> float:
    """把 chunks 逐段生成、逐段过闸门 1，返回按汉字数加权的覆盖率。

    每段单独转写而不是拼起来一次转写：跳读发生在**段内**，逐段读数才能定位到
    是哪一段丢了内容。

    判定走 `verify.check_content` 而不是自己调 ASR：语言与温度的钉法、双侧数字
    归一化都在那个函数里，探测若自己写一遍，量的就不是闸门而是它的仿制品。

    整体读数因此也由逐段读数加权合成，而不是再对全文做一次比较——那次比较会绕开
    闸门的归一化，把「一千二百 vs 1200」这类正字法差异算成内容丢失。加权而非算术
    平均，因为一个 11 汉字的标题段和一个 60 汉字的正文段对「丢了多少」贡献不同。
    """
    print(f"\n=== {label}：{len(chunks)} 段 ===")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    readings: list[tuple[int, float]] = []
    for index, chunk in enumerate(chunks, start=1):
        segment = synthesizer.generate(
            chunk, ref_audio=profile.anchor_wav, ref_text=profile.anchor_text
        )
        wav = PROBE_DIR / f"{label}-{index}.wav"
        wavio.write(wav, segment.audio, OUTPUT_SAMPLE_RATE)
        gate = check_content(wav, chunk, transcriber)

        han = len(textmatch.han_only(chunk))
        readings.append((han, gate.coverage))
        seconds_per_han = segment.audio_seconds / han if han else float("nan")
        flag = "" if gate.passed else "  ← 丢内容"
        print(
            f"  第 {index}/{len(chunks)} 段  {han:3d} 汉字  "
            f"覆盖率 {gate.coverage:7.2%}  {seconds_per_han:.4f} 秒每汉字{flag}"
        )

    total_han = sum(han for han, _ in readings)
    if not total_han:
        raise ValueError(f"{label}：全部切片都不含汉字，读数无意义")
    weighted = sum(han * value for han, value in readings) / total_han
    print(f"  加权覆盖率 {weighted:.2%}（{total_han} 汉字）")
    return weighted


def _thresholds(raw: str) -> tuple[int, ...]:
    """把 `900,80,60` 解析成阈值元组。切段阈值必须为正——0 会让每句都触发告警。"""
    values = tuple(int(part) for part in raw.split(","))
    if not values or any(value < 1 for value in values):
        raise ValueError(f"阈值必须是正整数，收到 {raw!r}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("post", type=Path, nargs="?", default=POST_DEFAULT, help="markdown 帖子")
    parser.add_argument(
        "--thresholds",
        type=_thresholds,
        default=THRESHOLDS_DEFAULT,
        help="要扫描的 threshold_han，逗号分隔",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not args.post.is_file():
        print(f"✗ 找不到探测用的帖子：{args.post}", file=sys.stderr)
        return 2

    spoken = _read_post(args.post)
    profile = load_profile(VOICE)
    synthesizer = Synthesizer()
    transcriber = load_transcriber()

    print(f"帖子 {args.post.name}")
    print(f"{len(textmatch.han_only(spoken))} 汉字，音色 {VOICE}")

    readings = {
        threshold: _speak_and_hear(
            f"阈值{threshold}",
            chunker.chunk(spoken, threshold_han=threshold),
            synthesizer,
            transcriber,
            profile,
        )
        for threshold in args.thresholds
    }

    print("\n---")
    for threshold, value in readings.items():
        print(f"  阈值 {threshold:4d}  加权覆盖率 {value:7.2%}")

    worst, best = readings[max(args.thresholds)], readings[min(args.thresholds)]
    if best - worst > 0.05:
        print("\n结论：跳读由单次生成的文本长度触发，CHUNK_THRESHOLD_HAN=900 定错了。")
    elif worst >= 0.97:
        print("\n结论：本次未复现失败——先确认音色与帖子是否与失败那次一致。")
    else:
        print("\n结论：切段没有解决跳读，原因在别处（锚点、标点或模型本身）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
