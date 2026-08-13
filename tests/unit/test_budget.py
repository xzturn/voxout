"""token 预算：把截断从「事后由闸门 1 发现」变成「事前由断言排除」。

生成循环最多跑 max_tokens 个 patch，每个 patch 产出固定时长的音频。
所以上限是可算的：patch_size × decode_chunk_size / sample_rate 秒/patch。
默认 max_tokens=2000 对应 320 秒 ≈ 1500 汉字——这就是设计 §11 里那个
「真实上限未知」的答案。
"""

import pytest

from voxout import config
from voxout.budget import BudgetError, plan_max_tokens, seconds_per_patch


class TestSecondsPerPatch:
    def test_voxcpm2_configuration_yields_the_documented_value(self) -> None:
        assert seconds_per_patch(4, 1920, 48000) == pytest.approx(
            config.SECONDS_PER_PATCH_NOMINAL
        )

    def test_zero_sample_rate_is_rejected_instead_of_dividing_by_zero(self) -> None:
        with pytest.raises(ValueError):
            seconds_per_patch(4, 1920, 0)

    def test_non_positive_patch_geometry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            seconds_per_patch(0, 1920, 48000)


class TestPlanMaxTokens:
    def test_real_post_length_gets_a_budget_with_margin(self) -> None:
        """572 汉字（真实短帖 2）：572 × 0.30 / 0.16 = 1072.5 → 1073 + 64 余量。"""
        assert plan_max_tokens(572, 0.16) == 1137

    def test_very_short_text_still_gets_the_fixed_margin(self) -> None:
        """短文本的相对误差最大，固定余量就是为它留的。"""
        assert plan_max_tokens(1, 0.16) >= config.TOKEN_MARGIN_PATCHES

    def test_budget_grows_monotonically_with_length(self) -> None:
        assert plan_max_tokens(100, 0.16) < plan_max_tokens(500, 0.16)

    def test_over_cap_is_refused_and_names_the_length_that_fits(self) -> None:
        """预算不够时正确的动作是切段，不是悄悄放大上限。

        错误消息必须给出能装下的汉字数，否则调用方只知道「不行」不知道「多少行」。
        """
        with pytest.raises(BudgetError) as excinfo:
            plan_max_tokens(10_000, 0.16)

        fits = int((config.MAX_TOKENS_CAP - config.TOKEN_MARGIN_PATCHES) * 0.16 / 0.30)
        assert str(fits) in str(excinfo.value)

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            plan_max_tokens(0, 0.16)

    def test_non_positive_patch_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            plan_max_tokens(100, 0.0)
