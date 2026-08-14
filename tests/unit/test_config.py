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


def test_voice_drift_threshold_clears_real_content_noise_and_still_catches_a_swap() -> None:
    """阈值必须高过真实语料的噪声底，且仍明显低于「换了个人」。

    上界侧的旧下界是 3.4%（设计 §4 的锚点方案实测），它放行了原来的 6%——而 6%
    低于自己估计器的噪声底，两篇真实短帖各出现一次误报。原因是 3.4% 与阈值的
    测量条件不同：3.4% 是**文本固定**下测的，阈值管的是文本各不相同的真实内容。
    换成同条件的读数，band 从 15 倍宽收到 6 倍宽：

        真实内容中位漂移最差 5.35% + 基线自身 n=3 的偏差约 3% ≈ 8.4%
        坏基线（零样本换人）48.82%
    """
    real_content_noise_floor_pct = 5.35 + 4.02 / 2
    bad_baseline_pct = 48.82
    assert real_content_noise_floor_pct < config.VOICE_DRIFT_MAX_PCT < bad_baseline_pct


def test_voice_root_is_absolute() -> None:
    assert config.VOICE_ROOT.is_absolute()
