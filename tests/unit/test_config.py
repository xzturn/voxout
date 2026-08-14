"""常量的自洽性。

单个常量没什么可测的；能出错的是常量之间的关系——两个机制各自看着合理，
合起来互相矛盾。
"""

from voxout import config


def test_coverage_threshold_is_a_fraction() -> None:
    assert 0.0 < config.COVERAGE_MIN < 1.0


def test_chunk_threshold_fits_inside_the_token_budget() -> None:
    """切段阈值必须落在 token 预算之内，否则两个防截断机制互相矛盾。

    一个刚好不触发切段的段落（`CHUNK_THRESHOLD_HAN` 汉字），其推算音频时长必须
    还在 token 预算能覆盖的范围里。否则会出现「不切段 + 预算不足」的死角，而这
    正是闸门 1 存在要防的那个失败模式。
    """
    longest_unchunked_seconds = config.CHUNK_THRESHOLD_HAN * config.SECONDS_PER_HAN_ESTIMATE
    budget_seconds = config.MAX_TOKENS_CAP * config.SECONDS_PER_PATCH_NOMINAL
    assert longest_unchunked_seconds < budget_seconds


def test_f0_search_range_is_ordered_and_covers_human_voice() -> None:
    assert config.F0_MIN_HZ < config.F0_MAX_HZ
    # 男低音约 80 Hz，女高音说话约 250 Hz，都要落在范围内。
    assert config.F0_MIN_HZ <= 80.0
    assert config.F0_MAX_HZ >= 250.0


def test_ltas_cutoff_within_reference_encoder_nyquist() -> None:
    """参考音在模型内部被降到 16kHz，Nyquist 8kHz。LTAS 截止不能超过它，
    否则锚点与输出在高频段比的是「重采样补出来的零」。"""
    reference_encoder_nyquist = 16000 / 2
    assert config.LTAS_MAX_HZ <= reference_encoder_nyquist


def test_voice_drift_threshold_sits_between_floor_and_bad_baseline() -> None:
    """6% 必须明确高于实测的锚点方案（3.4%）、明确低于坏基线（53.9%）。"""
    measured_anchor_spread_pct = 3.4
    bad_baseline_pct = 53.9
    assert measured_anchor_spread_pct < config.VOICE_DRIFT_MAX_PCT < bad_baseline_pct


def test_voice_root_is_absolute() -> None:
    assert config.VOICE_ROOT.is_absolute()
