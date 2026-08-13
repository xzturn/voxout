"""Can a voice ANCHOR pin identity across different texts?

Probe 1 proved naive zero-shot chunking is broken (F0 spread 41-45% across chunks)
and that a seed cannot fix it. This probe tests the three candidate fixes:

  E  ref_audio anchor  -- every chunk conditioned on one frozen reference clip
  F  continuation      -- each chunk conditioned on the PREVIOUS chunk's audio
  G  instruct          -- a text voice description instead of a reference clip

Pass criterion, set before running: to be usable for article synthesis the F0
spread across chunks must land near the within-voice jitter floor, not the
41-45% seen in probe 1. Anything above ~10% still means audibly different people.
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
ANCHOR_WAV = f"{OUT_DIR}/anchor.wav"
ANCHOR_TEXT = base.SAME_TEXT
ANCHOR_SEED = 42
INSTRUCT = "沉稳的中年男声，语速偏慢，像纪录片旁白"
TEXTS = base.DIFFERENT_TEXTS
PASS_THRESHOLD_PCT = 10.0


def stats_of(audio: np.ndarray) -> tuple[float, float, np.ndarray]:
    return base.median_f0(audio), base.spectral_centroid(audio), base.ltas(audio)


def summarize(name: str, clips: list[tuple[str, np.ndarray]],
              anchor: tuple[float, np.ndarray] | None = None) -> float:
    print(f"\n=== {name}", flush=True)
    rows = [(tag, *stats_of(a)) for tag, a in clips]
    for tag, f0, cen, spec in rows:
        extra = ""
        if anchor is not None:
            extra = f"   vs_anchor: dF0={f0 - anchor[0]:+6.1f} Hz  cos={base.cosine(spec, anchor[1]):.4f}"
        print(f"  {tag:<16} medianF0={f0:6.1f} Hz  centroid={cen:6.0f} Hz{extra}", flush=True)

    f0s = np.array([r[1] for r in rows])
    spread = float(np.nanmax(f0s) - np.nanmin(f0s))
    pct = 100 * spread / float(np.nanmean(f0s))
    print(f"  -> F0 spread across chunks: {spread:.1f} Hz ({pct:.1f}% of mean)  "
          f"[{'PASS' if pct <= PASS_THRESHOLD_PCT else 'FAIL'} vs {PASS_THRESHOLD_PCT}% bar]",
          flush=True)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            print(f"       LTAS {rows[i][0]} vs {rows[j][0]}: {base.cosine(rows[i][3], rows[j][3]):.4f}",
                  flush=True)
    return pct


def synth(model, tag: str, **kw) -> np.ndarray:
    t0 = time.perf_counter()
    results = list(model.generate(**kw))
    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
    print(f"  {tag:<16} {len(audio) / SR:5.2f}s audio  {time.perf_counter() - t0:5.2f}s wall",
          flush=True)
    wavfile.write(f"{OUT_DIR}/{tag}.wav", SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return audio


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load("mlx-community/VoxCPM2-8bit")

    # Freeze the anchor reproducibly: seed + text -> bit-identical clip (probe 1, row B).
    import mlx.core as mx
    mx.random.seed(ANCHOR_SEED)
    print("--- forging anchor (seed 42 + fixed text => reproducible) ---", flush=True)
    anchor_audio = synth(model, "anchor", text=ANCHOR_TEXT)
    anchor_f0, _, anchor_spec = stats_of(anchor_audio)
    print(f"  anchor medianF0 = {anchor_f0:.1f} Hz", flush=True)

    print("\n--- E: ref_audio anchor on every chunk ---", flush=True)
    e = [(f"E_ref_t{i}", synth(model, f"E_ref_t{i}", text=t,
                               ref_audio=ANCHOR_WAV, ref_text=ANCHOR_TEXT))
         for i, t in enumerate(TEXTS)]

    print("\n--- F: continuation chain (each chunk conditioned on previous) ---", flush=True)
    f_clips: list[tuple[str, np.ndarray]] = []
    prev_audio, prev_text = ANCHOR_WAV, ANCHOR_TEXT
    for i, t in enumerate(TEXTS):
        tag = f"F_cont_t{i}"
        audio = synth(model, tag, text=t, prompt_audio=prev_audio, prompt_text=prev_text)
        f_clips.append((tag, audio))
        prev_audio, prev_text = f"{OUT_DIR}/{tag}.wav", t

    print("\n--- G: instruct (voice design) on every chunk ---", flush=True)
    g = [(f"G_instr_t{i}", synth(model, f"G_instr_t{i}", text=t, instruct=INSTRUCT))
         for i, t in enumerate(TEXTS)]

    print(f"\nbaseline from probe 1 -> D (no anchor): 41.2%   C (seed only): 45.0%", flush=True)
    pct_e = summarize("E  ref_audio anchor", e, (anchor_f0, anchor_spec))
    pct_f = summarize("F  continuation chain", f_clips, (anchor_f0, anchor_spec))
    pct_g = summarize("G  instruct / voice design", g)

    print("\n=== verdict ===", flush=True)
    for label, pct in (("E ref_audio anchor", pct_e), ("F continuation", pct_f),
                       ("G instruct", pct_g)):
        print(f"  {label:<22} F0 spread {pct:5.1f}%   "
              f"{'PASS' if pct <= PASS_THRESHOLD_PCT else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
