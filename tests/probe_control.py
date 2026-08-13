"""Calibrate the metrics against controls before judging anything.

Probes 1-2 compared F0 spread to a threshold I invented (10%). That is exactly the
mistake lessons.md #1 warns about: asserting against a half-remembered bar instead
of an independent reference. So establish both ends of the scale first.

  POSITIVE CONTROL  one single continuous generation, split into thirds.
                    Voice is identical BY CONSTRUCTION (one utterance, one sampling
                    trajectory). Whatever spread this shows is the within-voice,
                    across-content jitter FLOOR -- prosody, not identity.

  NEGATIVE CONTROL  two probe-1 clips already known to be different speakers
                    (142 Hz vs 222 Hz). This is the CEILING for "same voice".

Only against those two can E/F/G be read. Also adds p10 F0 (the voiced-pitch floor),
which tracks speaker baseline pitch and is far less intonation-sensitive than median.
"""

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_audio.tts.utils import load
from scipy.io import wavfile

import probe_voice_consistency as base

SR = base.SR
OUT_DIR = "/tmp/wmprobe/voxprobe/out"
LONG_TEXT = "".join(base.DIFFERENT_TEXTS)


def voiced_f0s(signal: np.ndarray) -> np.ndarray:
    """All voiced-frame F0 estimates, not just the median."""
    lag_min = int(SR / base.F0_MAX_HZ)
    lag_max = int(SR / base.F0_MIN_HZ)
    windowed = base.frames_of(signal) * np.hanning(base.FRAME)
    centered = windowed - windowed.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * base.FRAME, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :lag_max + 1]
    zero = autocorr[:, :1]
    normalized = np.divide(autocorr, zero, out=np.zeros_like(autocorr), where=zero > 0)
    window = normalized[:, lag_min:lag_max + 1]
    peak_lag = window.argmax(axis=1) + lag_min
    peak_val = window.max(axis=1)
    energy = np.sqrt((windowed ** 2).mean(axis=1))
    voiced = (peak_val > base.VOICED_AUTOCORR_MIN) & (energy > 0.05 * energy.max())
    return SR / peak_lag[voiced] if voiced.any() else np.array([np.nan])


def descriptor(signal: np.ndarray) -> dict:
    f0s = voiced_f0s(signal)
    return {
        "median_f0": float(np.median(f0s)),
        "p10_f0": float(np.percentile(f0s, 10)),
        "ltas": base.ltas(signal),
    }


def spread_pct(values: list[float]) -> float:
    arr = np.array(values)
    return 100.0 * (np.nanmax(arr) - np.nanmin(arr)) / float(np.nanmean(arr))


def pairwise_cos(descs: list[dict]) -> list[float]:
    return [base.cosine(descs[i]["ltas"], descs[j]["ltas"])
            for i in range(len(descs)) for j in range(i + 1, len(descs))]


def evaluate(name: str, clips: list[tuple[str, np.ndarray]]) -> dict:
    descs = [descriptor(a) for _, a in clips]
    cos = pairwise_cos(descs)
    row = {
        "name": name,
        "median_spread": spread_pct([d["median_f0"] for d in descs]),
        "p10_spread": spread_pct([d["p10_f0"] for d in descs]),
        "cos_min": float(np.min(cos)),
        "cos_mean": float(np.mean(cos)),
    }
    detail = "  ".join(f"{tag.split('_')[-1]}:{d['median_f0']:.0f}/{d['p10_f0']:.0f}"
                       for (tag, _), d in zip(clips, descs))
    print(f"  {name:<26} medF0spread={row['median_spread']:5.1f}%  "
          f"p10spread={row['p10_spread']:5.1f}%  LTAScos min={row['cos_min']:.3f} "
          f"mean={row['cos_mean']:.3f}   [{detail}]", flush=True)
    return row


def read_wav(tag: str) -> np.ndarray:
    _, data = wavfile.read(f"{OUT_DIR}/{tag}.wav")
    return data.astype(np.float32) / 32768.0


def main() -> None:
    model = load("mlx-community/VoxCPM2-8bit")

    print("--- POSITIVE CONTROL: one continuous utterance, split into thirds ---", flush=True)
    t0 = time.perf_counter()
    results = list(model.generate(LONG_TEXT))
    long_audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
    print(f"  one call: {len(long_audio) / SR:.2f}s audio in {time.perf_counter() - t0:.2f}s "
          f"({len(LONG_TEXT)} chars, {len(results)} yields)", flush=True)
    wavfile.write(f"{OUT_DIR}/control_continuous.wav", SR,
                  (np.clip(long_audio, -1, 1) * 32767).astype(np.int16))
    third = len(long_audio) // 3
    pos_clips = [(f"pos_p{i}", long_audio[i * third:(i + 1) * third]) for i in range(3)]

    print("\n=== calibrated comparison ===", flush=True)
    print("  (medF0spread = median-pitch spread; p10spread = pitch-floor spread;\n"
          "   LTAScos = long-term spectrum cosine, higher = same timbre)\n", flush=True)

    rows = [
        evaluate("POS CONTROL same voice", pos_clips),
        evaluate("NEG CONTROL diff voices",
                 [(t, read_wav(t)) for t in ("D_noseed_t0", "D_noseed_t1", "D_noseed_t2")]),
        evaluate("C seed only", [(t, read_wav(t)) for t in
                                 ("C_seed42_t0", "C_seed42_t1", "C_seed42_t2")]),
        evaluate("E ref_audio anchor", [(t, read_wav(t)) for t in
                                        ("E_ref_t0", "E_ref_t1", "E_ref_t2")]),
        evaluate("F continuation chain", [(t, read_wav(t)) for t in
                                          ("F_cont_t0", "F_cont_t1", "F_cont_t2")]),
        evaluate("G instruct", [(t, read_wav(t)) for t in
                                ("G_instr_t0", "G_instr_t1", "G_instr_t2")]),
    ]

    pos, neg = rows[0], rows[1]
    print(f"\n  scale established by controls:", flush=True)
    print(f"    same voice  -> medF0spread {pos['median_spread']:.1f}%  "
          f"LTAScos_min {pos['cos_min']:.3f}", flush=True)
    print(f"    diff voices -> medF0spread {neg['median_spread']:.1f}%  "
          f"LTAScos_min {neg['cos_min']:.3f}", flush=True)

    print(f"\n=== verdict, judged against the controls ===", flush=True)
    for row in rows[2:]:
        # Position on the same-voice .. different-voice scale, via LTAS cosine.
        span = pos["cos_min"] - neg["cos_min"]
        score = (row["cos_min"] - neg["cos_min"]) / span if span > 0 else float("nan")
        print(f"  {row['name']:<24} LTAScos_min={row['cos_min']:.3f}  "
              f"-> {100 * score:5.1f}% of the way from DIFFERENT-voice to SAME-voice",
              flush=True)


if __name__ == "__main__":
    main()
