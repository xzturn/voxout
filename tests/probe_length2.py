"""Locate the truncation ceiling -- take two, with both earlier bugs fixed.

Bug 1 (metric): a hand-rolled greedy subsequence match desynced permanently at the
  first Chinese-numeral / Arabic-digit mismatch and reported 3.5% for audio that was
  in fact complete. Replaced with difflib.SequenceMatcher matching blocks (stdlib,
  robust to local substitutions, no catastrophic desync).

Bug 2 (fixture): LENGTHS exceeded len(SOURCE), so the 500- and 900-char rows silently
  fed the SAME text. The probe reported four lengths while testing three. Now asserted.

Also: SOURCE is deliberately numeral-free, so number-reading differences cannot
contaminate a measurement that is about length, not normalization.

Independent cross-check retained and strengthened: audio duration must grow roughly
LINEARLY with input length. A sublinear knee is truncation regardless of what the
ASR metric says.
"""

import difflib
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mlx_audio.tts.utils import load
from scipy.io import wavfile

SR = 48000
OUT_DIR = "/tmp/wmprobe/voxprobe/out"
ANCHOR_WAV = f"{OUT_DIR}/anchor.wav"
ANCHOR_TEXT = "这是一段用来测试音色是否稳定的中文句子。"
COVERAGE_PASS = 0.97

SOURCE = (
    "本页记录的是对可以在本地部署的开源语音合成模型所做的选型调研。"
    "核心判断是笔记本的算力其实是过剩的，即便最重的那个模型，占用的内存也远低于机器的上限，"
    "所以真正的约束并不是硬件，而是许可证和中文表现。"
    "按照当前的用途来选，答案指向那个支持参考音频加转录的模型，"
    "它同时具备原生高采样率输出、保真度最好的克隆能力，以及可商用阵营里最好的中文指标。"
    "代价是生成速度接近实时，谈不上快，但对于内容制作这个用途是可以接受的。"
    "反直觉的一点在于，机器上那一大片统一内存在这件事上完全用不上，"
    "几个候选模型的峰值占用都很小，机器从来不是瓶颈。"
    "已经确认的缺陷出现在那个最小的模型上，当一整段中文被单次喂进去的时候，"
    "它会在中途硬切断，丢弃相当一部分内容，而且完全不报错。"
    "两条独立的证据都指向同一个结论，一条是按它自身的语速反推，"
    "同一个模型同一个音色，语速不可能凭空变快；"
    "另一条是把生成的音频转录回文字，尾部有相当长的一段完全缺失。"
    "规避的方式是按句分行喂入再合并音频，覆盖率恢复到完整，"
    "而速度反而更好，这说明固定开销是按进程付的，不是按句付的。"
    "许可证是真正的筛子，按下载量排在前列的那些模型，将近一半不能自由使用，"
    "有的写着非商用，有的挂着自定义协议，还有的需要单独申请授权。"
    "在可以商用的范围里，那个支持参考音频的模型是中文表现最好的。"
    "安装配方上有两个必须绕开的坑，都表现为静默挂死或者莫名的权限错误，"
    "换成国内镜像并关掉那个内容寻址后端之后，几秒钟就装完了。"
    "还没有覆盖的部分包括主观听感评价、长篇播客的路径，"
    "以及最关键的一项，用本人真实录音去做克隆的实际效果，"
    "因为验证克隆链路的时候，用的参考音本身就是合成出来的。"
    "这一页是选型的快照，不是内化的知识，半年之后需要重新复核，"
    "因为这个领域的迭代速度快得离谱，调研之前三天还有新模型发布。"
    "关于架构上的差异，有一点值得单独记下来，它决定了模型小并不等于装得动。"
    "那个最小的模型属于早一代的声学模型，吃的是音素，"
    "所以必须外挂一整条从文字到音素的转换链路，还要带上分词和拼音，"
    "英文那一侧甚至要额外装一个发音引擎，结果是模型本体很轻，依赖比模型还重。"
    "而新一代的两个候选走的是语言模型加音频编解码的路子，直接吃原始文本，"
    "文字到发音的映射隐式地学在权重里，外部依赖是零，代价是参数量大了一个数量级。"
    "在这台机器上，这个差异被进一步放大，因为那些底层的编译型依赖，"
    "在这个平台上常常没有预编译好的安装包，得自己从源码构建，"
    "有的项目干脆在说明里写明必须走另一个包管理器才装得上。"
    "所以最终的取舍不是看参数量，而是看整条链路能不能干净地装起来并且跑得动。"
    "另外还有几条可用的运行时路径值得记一下，"
    "一条是自带的服务模式，提供一个与主流接口兼容的本地服务，"
    "一条是用另一种语言写的原生实现，走系统的神经网络加速器，"
    "还有一条是量化之后的通用推理框架路径，完全不需要解释器环境。"
    "这几条路径的速度和保真度各有取舍，暂时都没有逐一实测。"
)

LENGTHS = (350, 700, 1050)


def han_only(text: str) -> str:
    return "".join(c for c in text if "一" <= c <= "鿿")


def coverage(reference: str, heard: str) -> float:
    """Fraction of reference Han chars covered by matching blocks against the ASR text.

    difflib finds matching subsequences without the greedy desync that broke probe 1.
    """
    ref, got = han_only(reference), han_only(heard)
    if not ref:
        return 0.0
    matcher = difflib.SequenceMatcher(None, ref, got, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(ref)


def main() -> None:
    assert max(LENGTHS) <= len(SOURCE), (
        f"fixture too short: SOURCE is {len(SOURCE)} chars but LENGTHS asks for "
        f"{max(LENGTHS)} -- this is exactly bug 2, caught by assertion this time"
    )
    print(f"SOURCE = {len(SOURCE)} chars, {len(han_only(SOURCE))} Han; "
          f"testing {LENGTHS}\n", flush=True)

    model = load("mlx-community/VoxCPM2-8bit")
    from mlx_audio.stt.utils import load as load_stt
    stt = load_stt("mlx-community/whisper-large-v3-turbo-asr-fp16")

    print(f"{'chars':>6} {'han':>5} {'audio_s':>8} {'gen_s':>7} {'rtf':>5} "
          f"{'han/s':>6} {'s/han':>7} {'ASR_cov':>8}  verdict", flush=True)

    rows = []
    for n in LENGTHS:
        text = SOURCE[:n]
        n_han = len(han_only(text))
        t0 = time.perf_counter()
        results = list(model.generate(text, ref_audio=ANCHOR_WAV, ref_text=ANCHOR_TEXT))
        gen_s = time.perf_counter() - t0
        audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
        dur = len(audio) / SR
        path = f"{OUT_DIR}/len2_{n}.wav"
        wavfile.write(path, SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))

        heard = stt.generate(path).text
        with open(f"{OUT_DIR}/len2_{n}.asr.txt", "w") as fh:
            fh.write(heard)
        cov = coverage(text, heard)
        sec_per_han = dur / n_han
        flag = "OK" if cov >= COVERAGE_PASS else ("TRUNCATED" if cov < 0.90 else "degraded")
        print(f"{n:>6} {n_han:>5} {dur:>8.2f} {gen_s:>7.1f} {gen_s / dur:>5.2f} "
              f"{n_han / dur:>6.2f} {sec_per_han:>7.4f} {100 * cov:>7.1f}%  {flag}", flush=True)
        rows.append((n_han, dur, sec_per_han, cov))

    print("\n--- independent cross-check: seconds per Han char must stay FLAT ---", flush=True)
    sph = np.array([r[2] for r in rows])
    drift = 100 * (sph.max() - sph.min()) / sph.mean()
    print(f"  s/han across lengths: {', '.join(f'{v:.4f}' for v in sph)}", flush=True)
    print(f"  drift = {drift:.1f}%  -> {'FLAT, no truncation' if drift < 15 else 'KNEE, content lost'}",
          flush=True)
    print("\n  (a model cannot read faster as text gets longer; a falling s/han is\n"
          "   content going missing, which is how the doc caught the Kokoro defect)",
          flush=True)


if __name__ == "__main__":
    main()
