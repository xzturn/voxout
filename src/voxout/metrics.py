"""音色度量原语：F0 / p10 / LTAS / 余弦 / 极差。

由本轮的探测脚本（tests/probe_control.py、probe_voice_consistency.py）升格而来，
一处改动与原探测不同：**采样率是显式参数，不是模块常量**。

理由：参考音在模型内部被降采样到 16 kHz，而我们的锚点是 48 kHz 文件。若把
44.1 kHz 的锚点按 48 kHz 解读，F0 会系统性偏高 8.8%——刚好吃掉闸门 2 的
6% 余量，表现为「音色漂移」而真实原因是单位错。硬编码采样率会把这个 bug
永久藏起来。

两个判据的分工：
  - **p10 F0**（浊音音高地板）是主判据。同一说话人的地板极稳（实测 0.9%），
    而中位数被语调污染（同一人 5.1%）。
  - **LTAS 余弦**是旁证。它是频谱包络的指纹，对音高几乎不敏感
    （200 Hz vs 250 Hz 实测 0.990），不能单独当音色判据。
"""

import numpy as np

from voxout.config import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    FRAME,
    HOP,
    LTAS_MAX_HZ,
    VOICED_AUTOCORR_MIN,
)

_ENERGY_FLOOR_RATIO = 0.05
"""帧能量低于全局峰值这个比例即视为静音。挡掉句间停顿被当浊音分析。"""


def frames_of(signal: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """切成重叠分析帧，返回 (n_frames, frame)。

    短于一帧的信号补零到一帧——否则索引会越界（原探测脚本的潜在 bug，
    因为探测用的音频都远长于一帧而从未暴露）。
    """
    if signal.ndim != 1:
        raise ValueError(f"只接受单声道一维信号，收到 shape={signal.shape}")

    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))

    n_frames = 1 + (len(signal) - frame) // hop
    offsets = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    return signal[offsets]


def voiced_f0s(signal: np.ndarray, sr: int) -> np.ndarray:
    """所有浊音帧的 F0 估计（归一化自相关法）。无浊音帧时返回空数组。

    自相关在频域算：rfft → 乘共轭 → irfft，比时域卷积快一个数量级。
    搜索窗限制在 [sr/F0_MAX, sr/F0_MIN] 延迟内，这个窗的边界直接决定了
    八度错误的发生位置（见 config.F0_MAX_HZ 的说明）。
    """
    lag_min, lag_max = int(sr / F0_MAX_HZ), int(sr / F0_MIN_HZ)
    windowed = frames_of(signal) * np.hanning(FRAME)
    centered = windowed - windowed.mean(axis=1, keepdims=True)

    spectrum = np.fft.rfft(centered, n=2 * FRAME, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, : lag_max + 1]
    zero_lag = autocorr[:, :1]
    # 静音帧的零延迟项为 0，除法必须被短路而不是产生 inf/nan。
    normalized = np.divide(autocorr, zero_lag, out=np.zeros_like(autocorr), where=zero_lag > 0)

    search_window = normalized[:, lag_min : lag_max + 1]
    peak_lag = search_window.argmax(axis=1) + lag_min
    peak_value = search_window.max(axis=1)

    energy = np.sqrt((windowed**2).mean(axis=1))
    is_voiced = (peak_value > VOICED_AUTOCORR_MIN) & (
        energy > _ENERGY_FLOOR_RATIO * energy.max()
    )
    return sr / peak_lag[is_voiced] if is_voiced.any() else np.array([])


def p10_f0(signal: np.ndarray, sr: int) -> float:
    """浊音音高地板（第 10 百分位）。闸门 2 的主判据。"""
    f0s = voiced_f0s(signal, sr)
    return float(np.percentile(f0s, 10)) if f0s.size else float("nan")


def median_f0(signal: np.ndarray, sr: int) -> float:
    """浊音音高中位数。**不可用作闸门判据**（被语调污染，设计 §4），仅供诊断对照。"""
    f0s = voiced_f0s(signal, sr)
    return float(np.median(f0s)) if f0s.size else float("nan")


def ltas(signal: np.ndarray, sr: int) -> np.ndarray:
    """长时平均频谱（dB），截止到 LTAS_MAX_HZ。"""
    windowed = frames_of(signal) * np.hanning(FRAME)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1)).mean(axis=0)
    n_bins = int(LTAS_MAX_HZ / (sr / FRAME))
    # +1e-10 防止静音频段取 log 得到 -inf，污染后续的余弦。
    return 20.0 * np.log10(magnitude[:n_bins] + 1e-10)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """去均值余弦（等价 Pearson 相关）。

    去均值是关键：dB 尺度的 LTAS 有很大的公共直流分量（整体音量），
    不去均值的话任意两段人声余弦都在 0.99 以上，判别力归零。

    退化输入（常数谱，方差为 0）返回 nan——相关性在此处无定义，
    返回 1.0 会让闸门误判通过。
    """
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denominator == 0:
        return float("nan")
    return float(a_centered @ b_centered / denominator)


def spread_pct(values: list[float]) -> float:
    """极差占均值的百分比。分母是均值，与设计 §4 的标尺同一定义。

    Raises:
        ValueError: 序列为空或全为 nan。全 nan 意味着没有任何浊音帧可比，
            静默返回 0% 会让闸门 2 假装通过。
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.isnan(array).all():
        raise ValueError("没有可比较的有效数值：序列为空或全为 nan")

    return 100.0 * float(np.nanmax(array) - np.nanmin(array)) / float(np.nanmean(array))
