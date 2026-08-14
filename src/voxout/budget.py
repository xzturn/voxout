"""token 预算：把截断从事后发现变成事前排除。

生成循环最多跑 `max_tokens` 个 patch，每个 patch 产出固定时长的音频：

    patch_size × decode_chunk_size / sample_rate 秒

VoxCPM2 是 4 × 1920 / 48000 = 0.16 秒/patch。默认 max_tokens=2000 → 320 秒
→ 按实测 0.2127 秒/汉字约 1500 汉字。设计 §11 的「真实上限未知」由此关闭。

选型文档记录过 Kokoro 静默截断（173 字输入在第 115 字处丢了 27%）。那种失败
之所以危险是因为它无声。这个模块的作用就是让同类失败在生成之前就报错。
"""

import math

from voxout.config import MAX_TOKENS_CAP, SECONDS_PER_HAN_ESTIMATE, TOKEN_MARGIN_PATCHES


class BudgetError(ValueError):
    """文本长度超出 token 预算上限。正确的处置是切段，不是放大上限。"""


def seconds_per_patch(patch_size: int, decode_chunk_size: int, sample_rate: int) -> float:
    """从模型配置推出每个 patch 的音频时长。运行时一律用这个，不用常量。"""
    if patch_size <= 0 or decode_chunk_size <= 0 or sample_rate <= 0:
        raise ValueError(
            f"patch 几何参数必须为正：patch_size={patch_size}、"
            f"decode_chunk_size={decode_chunk_size}、sample_rate={sample_rate}"
        )
    return patch_size * decode_chunk_size / sample_rate


def plan_max_tokens(n_han: int, sec_per_patch: float) -> int:
    """按汉字数推算 max_tokens。

    宁可高估：多给预算没有成本（真正的停止由模型的 stop head 决定），
    低估则直接导致截断。

    Raises:
        BudgetError: 需要的预算超过 MAX_TOKENS_CAP。消息含能装下的汉字数，
            让调用方知道「多少行」而不只是「不行」。
        ValueError: 参数非正。
    """
    if n_han <= 0:
        raise ValueError(f"汉字数必须为正，收到 {n_han}")
    if sec_per_patch <= 0:
        raise ValueError(f"每 patch 时长必须为正，收到 {sec_per_patch}")

    needed = (
        math.ceil(n_han * SECONDS_PER_HAN_ESTIMATE / sec_per_patch) + TOKEN_MARGIN_PATCHES
    )
    if needed > MAX_TOKENS_CAP:
        usable_patches = MAX_TOKENS_CAP - TOKEN_MARGIN_PATCHES
        fits = int(usable_patches * sec_per_patch / SECONDS_PER_HAN_ESTIMATE)
        raise BudgetError(
            f"{n_han} 汉字需要 {needed} 个 patch，超过上限 {MAX_TOKENS_CAP}。"
            f"单段请控制在 {fits} 汉字以内，或降低切段阈值让 chunker 分段。"
        )

    return needed
