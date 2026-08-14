"""命令行入口：两个子命令、三个退出码、带修复命令的失败消息。

退出码是这一层真正的契约——它决定这套东西能不能进脚本：

  0  成功。音色闸门可能只是告警：它的判据是代理指标，不该挡住流程
  1  内容闸门不过。**预期内**的失败，看过差异报告后重跑或 --force
  2  操作性错误：档案缺失、输入为空、权重没缓存。argparse 的用法错误同码

两个昂贵资源（3.0GB TTS 权重 + ASR 权重）由工厂注入，单测因此完全不碰模型。
`render` 把档案校验排在构造合成器之前：音色名拼错是最常见的失误，而
`load_profile` 是免费的。`mint-voice` 的「已存在」检查留在 mint_profile 内部、
模型加载之后——一个音色一生只铸一次，那 2 秒不值得把校验逻辑抄第二遍。

离线默认**不在这里**：它在包的 `__init__`，因为必须早于任何 huggingface_hub
导入，而那可能发生在任何一次 `import voxout.x`，不只是走 CLI 的时候。

人看的报告用 print——stdout/stderr 就是 CLI 的接口；库模块一律用 logging，
main 把它接到 INFO，这样渲染进度与库的告警都能到人眼前。
"""

import argparse
import logging
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from voxout.config import ANCHOR_TEXT_DEFAULT, VOICE_ROOT
from voxout.profile import VoiceProfile, load_profile, mint_profile
from voxout.render import RenderResult, render
from voxout.synth import Synthesizer
from voxout.verify import ContentGate, TranscriberLike, VoiceGate, load_transcriber

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2

_MINT = "mint-voice"
_RENDER = "render"

# 工厂而非实例：昂贵资源必须在参数与档案校验之后才构造。单测传替身即可——
# 两个子命令用到的方法在 Synthesizer 上都有，鸭子类型足够。
SynthesizerFactory = Callable[[], Synthesizer]
TranscriberFactory = Callable[[], TranscriberLike]


def _configure_logging() -> None:
    """把库的日志接到 stderr。

    basicConfig 在已有 handler 时是空操作，所以这行不会抢走调用方（或 pytest）
    已经配好的日志。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _add_voice_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--voice-root",
        type=Path,
        default=VOICE_ROOT,
        help=f"音色档案根目录（默认 {VOICE_ROOT}）",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxout", description="把 markdown 渲染成音色固定的 48kHz 对外音频"
    )
    subcommands = parser.add_subparsers(dest="command")

    mint = subcommands.add_parser(_MINT, help="零样本生成一段音频，固定为音色锚点")
    mint.add_argument("name", help="音色名，例如 house")
    mint.add_argument(
        "--text",
        default=ANCHOR_TEXT_DEFAULT,
        help="锚点文本。默认是设计 §4 标定用的那一句，换文本需重新标定闸门 2",
    )
    mint.add_argument("--seed", type=int, default=None, help="生成种子，给出才可复现")
    _add_voice_root(mint)

    render_command = subcommands.add_parser(_RENDER, help="渲染一篇 markdown")
    render_command.add_argument("source", type=Path, help="markdown 源文件")
    render_command.add_argument("--voice", required=True, help="音色名")
    render_command.add_argument(
        "-o", "--output", type=Path, default=None, help="输出 wav（默认 out/<帖子名>.wav）"
    )
    render_command.add_argument(
        "--force",
        action="store_true",
        help="内容闸门不过也产出音频，并在 manifest 里标记 forced",
    )
    _add_voice_root(render_command)

    return parser


def _metric(manifest: dict[str, object], key: str) -> float:
    """manifest 的值是 object（它要能序列化成 JSON）。取不到数就给 nan——
    显式打印 nan 比崩掉或打印 0 好：0 会被当成真实读数。"""
    value = manifest[key]
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _content_line(gate: ContentGate) -> str:
    verdict = "通过" if gate.passed else "未通过"
    note = "  ⚠ ASR 数字归一化失败，覆盖率可能偏低" if gate.asr_normalize_failed else ""
    return (
        f"  内容闸门  覆盖率 {gate.coverage:.2%}"
        f"（阈值 {gate.threshold:.0%}）  {verdict}{note}"
    )


def _voice_line(gate: VoiceGate) -> str:
    verdict = "通过" if gate.passed else "未通过"
    return (
        f"  音色闸门  漂移 {gate.drift_pct:.2f}%"
        f"（上限 {gate.threshold:.0f}%）  {verdict}  段间极差 {gate.spread_pct:.2f}%"
    )


def _timing_line(manifest: dict[str, object]) -> str:
    return (
        f"  时长 {_metric(manifest, 'audio_seconds'):.1f} 秒 / "
        f"{_metric(manifest, 'han_chars'):.0f} 汉字  "
        f"{_metric(manifest, 'seconds_per_han'):.4f} 秒每汉字  "
        f"生成耗时 {_metric(manifest, 'generation_seconds'):.1f} 秒"
    )


def _force_hint(result: RenderResult) -> str:
    source = shlex.quote(str(result.source))
    return f"    uv run voxout render {source} --voice {result.voice} --force"


def _report_content_failure(result: RenderResult) -> None:
    """闸门 1 不过且未 --force：没有音频，只有证据。

    消息要回答人接下来的三个问题：差多少、去哪看、怎么强制出。
    """
    gate = result.content_gate
    print(
        f"✗ 内容闸门未通过  覆盖率 {gate.coverage:.2%}（阈值 {gate.threshold:.0%}）",
        file=sys.stderr,
    )
    print(f"  差异报告  {result.diff_path}", file=sys.stderr)
    print(
        "  未产出音频：一个丢了内容的音频比没有文件更危险，它看起来像成品。",
        file=sys.stderr,
    )
    print("  逐字比对确认是 ASR 听错而非真丢内容后，可强制产出：", file=sys.stderr)
    print(_force_hint(result), file=sys.stderr)


def _report(result: RenderResult) -> int:
    if result.audio_path is None:
        _report_content_failure(result)
        return EXIT_GATE_FAILED

    print(f"✓ 渲染完成  {result.audio_path}")
    print(_content_line(result.content_gate))
    if result.voice_gate is not None:
        print(_voice_line(result.voice_gate))
    if result.manifest is not None:
        print(_timing_line(result.manifest))
        print(f"  manifest  {result.manifest_path}")

    if not result.content_gate.passed:
        print(
            f"⚠ 内容闸门未通过，但已按 --force 产出，manifest 记为 forced=true。"
            f"差异报告：{result.diff_path}",
            file=sys.stderr,
        )
    if result.voice_gate is not None and not result.voice_gate.passed:
        print(
            f"⚠ 音色闸门未通过。判据是代理指标（p10 F0），可能误报——"
            f"请试听 {result.audio_path} 再决定是否重出。",
            file=sys.stderr,
        )
    return EXIT_OK


def _report_mint(profile: VoiceProfile) -> None:
    print(f"✓ 铸造完成  {profile.name}")
    print(f"  锚点音频  {profile.anchor_wav}")
    print(f"  锚点文本  {profile.anchor_text}")
    print(f"  基线 p10 F0  {profile.baseline_p10_f0:.2f} Hz（闸门 2 漂移的分母）")
    seed = profile.seed if profile.seed is not None else "未指定（这次铸造不可复现）"
    print(f"  种子  {seed}")


def _mint_command(args: argparse.Namespace, make_synthesizer: SynthesizerFactory) -> int:
    profile = mint_profile(
        args.name,
        args.text,
        make_synthesizer(),
        seed=args.seed,
        root=args.voice_root,
    )
    _report_mint(profile)
    return EXIT_OK


def _render_command(
    args: argparse.Namespace,
    make_synthesizer: SynthesizerFactory,
    make_transcriber: TranscriberFactory,
) -> int:
    profile = load_profile(args.voice, root=args.voice_root)
    result = render(
        args.source,
        profile,
        make_synthesizer(),
        make_transcriber(),
        output=args.output,
        forced=args.force,
    )
    return _report(result)


def main(
    argv: list[str] | None = None,
    *,
    synthesizer_factory: SynthesizerFactory = Synthesizer,
    transcriber_factory: TranscriberFactory = load_transcriber,
) -> int:
    """解析参数、分派子命令，把异常翻译成退出码。

    只捕获这三类：档案与文件问题（OSError）、校验与预算问题（ValueError，
    BudgetError 是它的子类）、模型加载问题（RuntimeError）。不捕获裸 Exception——
    别的异常是 bug，应该带栈退出而不是被压成一行「✗ ...」。
    """
    _configure_logging()
    parser = _parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_ERROR

    try:
        if args.command == _MINT:
            return _mint_command(args, synthesizer_factory)
        return _render_command(args, synthesizer_factory, transcriber_factory)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
