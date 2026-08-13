"""全部常量集中于此。每个阈值都注明它来自哪次测量或设计的哪一节。

没有来源的阈值就是猜的，而猜的阈值会在第一次误报后被关掉——等于没有闸门。
"""

from pathlib import Path

# ---------- 模型 ----------

MODEL_ID = "mlx-community/VoxCPM2-8bit"
"""可商用范围内中文表现最好的（选型文档 §3）。8bit 权重 3.0GB，峰值内存 3.2GB。"""

ASR_MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"
"""闸门 1 的回环转录器。"""

OUTPUT_SAMPLE_RATE = 48000
"""VoxCPM2 原生输出率，实测确认（设计 §3）。锚点也要求这个率，理由见 profile.py。"""

SECONDS_PER_PATCH_NOMINAL = 0.16
"""patch_size(4) × decode_chunk_size(1920) / 48000。仅用于常量自洽性检查与文档；
运行时一律从模型配置派生（synth.Synthesizer.seconds_per_patch），不信这个数。"""

# ---------- 度量原语 ----------

FRAME = 2048
HOP = 1024

F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
"""人声基频搜索范围。超出会发生八度错误（440 Hz 正弦被判 220.2 Hz）——这是取值的
后果而非缺陷，已写成断言（tests/unit/test_metrics.py），防止日后被当 bug「修」掉。"""

VOICED_AUTOCORR_MIN = 0.30
"""判定浊音帧的归一化自相关峰值下限。"""

LTAS_MAX_HZ = 8000.0
"""长时平均频谱的截止。参考音在模型内部被降到 16kHz（Nyquist 8kHz），取此值使
锚点与 48kHz 输出的 LTAS 在同一带宽上可比。"""

# ---------- 两道闸门 ----------

COVERAGE_MIN = 0.97
"""闸门 1：ASR 汉字覆盖率下限。实测三个长度稳定在 98.4–98.9%（设计 §3），留约
1.5 个百分点余量。注意本轮改了 ASR 参数并加了双侧归一化，阈值在 T13 重新标定。"""

VOICE_DRIFT_MAX_PCT = 6.0
"""闸门 2：p10 F0 相对锚点基线的漂移上限。标尺（设计 §4）：同一人地板 0.9%，
锚点方案实测 3.4%，坏基线 53.9%。6% 留一倍余量，且距坏区一个数量级。"""

# ---------- 文本与预算 ----------

CHUNK_THRESHOLD_HAN = 900
"""超过这个汉字数才切段。实测 1050 汉字零截断，900 是保守值。"""

SECONDS_PER_HAN_ESTIMATE = 0.30
"""每汉字音频秒数的保守上界。实测 0.2114–0.2142（设计 §6），留 40% 余量。
只用于事前推算 token 预算，宁可高估（多给预算无成本，停止由模型的 stop head 决定）。"""

TOKEN_MARGIN_PATCHES = 64
"""token 预算的固定余量，吸收短文本的相对误差。"""

MAX_TOKENS_CAP = 4000
"""token 预算硬上限。默认 2000 对应 320 秒音频；超过 4000 的内存与质量未实测，
故拒绝而非放行——预算不够时应该切段，不是悄悄放大。"""

CHUNK_GAP_SECONDS = 0.2
"""多段拼接时的段间静音。需人工听审确认（docs/listening-review.md）。"""

# ---------- 路径 ----------

VOICE_ROOT = Path.home() / "Life OS/60-69 Content Production/68 Writing Assets/voice"
"""音色档案根目录（设计 §5）。每个音色一个子目录：anchor.wav / anchor.txt / profile.toml。"""

OUTPUT_DIR = Path("out")
"""渲染产物默认目录，已在 .gitignore 中。"""
