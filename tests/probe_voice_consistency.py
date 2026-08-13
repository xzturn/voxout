"""Quantify VoxCPM2 zero-shot voice consistency across calls.

The selection doc measured RTF / memory / intelligibility with ONE generation per
model. It never asked: does zero-shot mode return the SAME speaker on every call?
For chunked article synthesis that is the load-bearing property.

Two independent discriminators, so a single flaky metric cannot fool us:
  1. median F0 (autocorrelation)  -- gross speaker identity; male/female differ ~2x
  2. LTAS cosine similarity       -- timbre fingerprint (long-term average spectrum)

Scenarios:
  A  same text, no seed control      -> is voice stable at all?
  B  same text, mx.random.seed(42)   -> is generation reproducible?
  C  different texts, same seed 42   -> DECISIVE: can a seed pin a voice across chunks?
  D  different texts, no seed        -> the naive chunked-article baseline
"""

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load
from scipy.io import wavfile

SR = 48000
FRAME = 2048
HOP = 1024
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
VOICED_AUTOCORR_MIN = 0.30
LTAS_MAX_HZ = 8000.0
OUT_DIR = "/tmp/wmprobe/voxprobe/out"

SAME_TEXT = "这是一段用来测试音色是否稳定的中文句子。"
DIFFERENT_TEXTS = (
    "第一段说的是选型的结论和它的代价。",
    "第二段解释为什么机器性能不是真正的瓶颈。",
    "第三段留给尚未验证的风险和后续计划。",
)


def frames_of(signal: np.ndarray) -> np.ndarray:
    """Split into overlapping analysis frames. Returns (n_frames, FRAME)."""
    n = 1 + max(0, (len(signal) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return signal[idx]


def median_f0(signal: np.ndarray) -> float:
    """Median F0 over voiced frames via normalized autocorrelation."""
    lag_min = int(SR / F0_MAX_HZ)
    lag_max = int(SR / F0_MIN_HZ)
    windowed = frames_of(signal) * np.hanning(FRAME)
    centered = windowed - windowed.mean(axis=1, keepdims=True)

    spectrum = np.fft.rfft(centered, n=2 * FRAME, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :lag_max + 1]
    zero_lag = autocorr[:, :1]
    normalized = np.divide(autocorr, zero_lag, out=np.zeros_like(autocorr), where=zero_lag > 0)

    window = normalized[:, lag_min:lag_max + 1]
    peak_lag = window.argmax(axis=1) + lag_min
    peak_val = window.max(axis=1)

    energy = np.sqrt((windowed ** 2).mean(axis=1))
    voiced = (peak_val > VOICED_AUTOCORR_MIN) & (energy > 0.05 * energy.max())
    if not voiced.any():
        return float("nan")
    return float(np.median(SR / peak_lag[voiced]))


def ltas(signal: np.ndarray) -> np.ndarray:
    """Long-term average spectrum in dB, truncated to LTAS_MAX_HZ."""
    windowed = frames_of(signal) * np.hanning(FRAME)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1)).mean(axis=0)
    n_bins = int(LTAS_MAX_HZ / (SR / FRAME))
    return 20.0 * np.log10(magnitude[:n_bins] + 1e-10)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_c, b_c = a - a.mean(), b - b.mean()
    return float(a_c @ b_c / (np.linalg.norm(a_c) * np.linalg.norm(b_c)))


def spectral_centroid(signal: np.ndarray) -> float:
    windowed = frames_of(signal) * np.hanning(FRAME)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    return float((freqs * magnitude).sum() / magnitude.sum())


def synth(model, text: str, seed: int | None, tag: str) -> np.ndarray:
    if seed is not None:
        mx.random.seed(seed)
    t0 = time.perf_counter()
    results = list(model.generate(text))
    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
    rtf = results[-1].real_time_factor
    print(
        f"  {tag:<14} {len(audio) / SR:5.2f}s audio  {time.perf_counter() - t0:5.2f}s wall  "
        f"lib_rtf={rtf:.2f}  yields={len(results)}",
        flush=True,
    )
    wavfile.write(f"{OUT_DIR}/{tag}.wav", SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return audio


def report(name: str, question: str, clips: list[tuple[str, np.ndarray]]) -> None:
    print(f"\n=== {name}: {question}", flush=True)
    stats = [(tag, median_f0(a), spectral_centroid(a), ltas(a)) for tag, a in clips]
    for tag, f0, cen, _ in stats:
        print(f"  {tag:<14} medianF0={f0:6.1f} Hz   centroid={cen:6.0f} Hz", flush=True)

    f0s = np.array([s[1] for s in stats])
    spread = float(np.nanmax(f0s) - np.nanmin(f0s))
    print(f"  -> F0 spread across calls: {spread:.1f} Hz "
          f"({100 * spread / np.nanmean(f0s):.1f}% of mean)", flush=True)

    print("  -> LTAS cosine similarity (timbre):", flush=True)
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            print(f"       {stats[i][0]} vs {stats[j][0]}: {cosine(stats[i][3], stats[j][3]):.4f}",
                  flush=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()
    model = load("mlx-community/VoxCPM2-8bit")
    print(f"model loaded in {time.perf_counter() - t0:.2f}s (this cost is PER PROCESS)\n", flush=True)

    print("--- generating ---", flush=True)
    a = [(f"A_noseed_{i}", synth(model, SAME_TEXT, None, f"A_noseed_{i}")) for i in range(3)]
    b = [(f"B_seed42_{i}", synth(model, SAME_TEXT, 42, f"B_seed42_{i}")) for i in range(3)]
    c = [(f"C_seed42_t{i}", synth(model, t, 42, f"C_seed42_t{i}"))
         for i, t in enumerate(DIFFERENT_TEXTS)]
    d = [(f"D_noseed_t{i}", synth(model, t, None, f"D_noseed_t{i}"))
         for i, t in enumerate(DIFFERENT_TEXTS)]

    report("A", "same text, NO seed -- is zero-shot voice stable?", a)

    print("\n=== B: same text + same seed -- bit-identical?", flush=True)
    n = min(len(b[0][1]), len(b[1][1]), len(b[2][1]))
    print(f"  lengths: {[len(x[1]) for x in b]}", flush=True)
    print(f"  max|B0-B1| over first {n} samples: {np.abs(b[0][1][:n] - b[1][1][:n]).max():.2e}",
          flush=True)
    print(f"  max|B0-B2| over first {n} samples: {np.abs(b[0][1][:n] - b[2][1][:n]).max():.2e}",
          flush=True)
    report("B", "same text + same seed -- F0/timbre view", b)

    report("C", "DIFFERENT texts, SAME seed -- can a seed pin the voice?", c)
    report("D", "DIFFERENT texts, no seed -- naive chunked-article baseline", d)

    print(f"\nwavs in {OUT_DIR}/ (listen to confirm the numbers)", flush=True)


if __name__ == "__main__":
    main()
