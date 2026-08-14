# voxout 实现计划

> **For agentic workers:** 用 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 勾选。
> 本计划为同会话即时执行而写：接口、阈值、测试断言全部给定；平凡模块给签名与行为规约，微妙处给完整代码。

**Goal:** 把 `63 Short-form Posts/*.md` 渲染成音色固定、内容完整、可对外发布的 48kHz 音频。

**Architecture:** 库内核 + 瘦 CLI，不起服务。纯文本模块（markdown / normalize / textmatch / chunker / budget / metrics）不碰模型，可秒级单测；碰模型的模块（synth / profile.mint / verify.gate1）走 `integration` 标记。一次渲染只加载一个模型实例。

**Tech Stack:** Python 3.12 · uv · mlx-audio 0.4.8（MLX 0.32.0）· VoxCPM2-8bit · whisper-large-v3-turbo · cn2an 0.5.24 · numpy/scipy · pytest

**依据:** [设计文档](../specs/2026-08-13-voxcpm2-content-tts-design.md)（已批准）

> **状态：13 个任务全部实现并提交**（`c35b887`..HEAD，另有 T13a/b/c 三次修正）。执行期间没有逐步勾选下面的 `- [ ]`，完成情况以 git 历史与设计文档的回填为准。实测读数（两道闸门的标定、秒/汉字 旁证、正负控制标尺）已写回设计 §3/§6/§10/§11；渲染产物在 `out/`（gitignore，不入库），其读数已转录进文档。人工听审待办见 [`docs/listening-review.md`](../../listening-review.md)。

## Global Constraints

- 语言 **Python**，不用 Rust。理由见设计 §2；本轮核实的补充：`mlx-rs` crate 存在（v0.25.3）但停更于 2025-12 且无模型库，candle/mlx-rs 均无 VoxCPM2 实现；决定性理由是 RTF≈0.93 下编排层占比 <0.1%，且两道闸门（whisper ASR + numpy F0）同样钉在 Python 上。
- 输出采样率固定 **48000 Hz**，int16 PCM WAV。
- 模型 `mlx-community/VoxCPM2-8bit`；ASR `mlx-community/whisper-large-v3-turbo-asr-fp16`。
- 闸门 1：ASR 汉字覆盖率 **≥ 0.97**，不过则**不产出音频**，`--force` 为逃生口。
- 闸门 2：p10 F0 相对锚点漂移 **≤ 15.0%**，取**各段漂移绝对值的中位数**；不过则告警 + manifest 标记，仍产出音频。（原为 ≤6.0% 取最大值，T13c 按真实语料重标定，理由见 `verify.check_voice` 与设计 §6 的回填。）
- **闸门 2 必须用 p10 F0，不得用中位数 F0**（设计 §4：中位数 F0 被语调污染，同一人天然 5.1% 极差）。
  ⚠️ 这一条与上一条不矛盾，两个「中位数」在不同的轴上：**段内**把 F0 分布折成一个数时必须取 p10（低端分位）而非中位数；**跨段**把各段的漂移折成一个判据时取中位数而非最大值。段内取中位数会被语调污染，跨段取最大值会随段数上升。
- F0 估计器有效范围 **60–400 Hz**，超出发生八度错误（440 Hz → 220.2 Hz）。这是取值后果，必须写成断言，防止日后被当 bug「修」掉。
- 所有渲染默认 `HF_HUB_OFFLINE=1`（权重已缓存 3.0GB）。宁可离线明确失败，不要静默挂死。
- 音色档案位置：`~/Life OS/60-69 Content Production/68 Writing Assets/voice/<name>/`，含 `anchor.wav` / `anchor.txt` / `profile.toml`。
- 切段阈值 **60 汉字**，且**段落原子**：段落之间一律分开，阈值只管单个超长段落内部要不要再切。（原为 900 汉字单次调用，T13a 实测单次长调用会跳读：213 汉字覆盖率 55.40%。）本轮两篇真实短帖清洗后为 213 / 540 汉字，分成 6 / 14 段。
- 不可变：数据类一律 `frozen=True`，函数返回新对象。
- 文件 200–400 行为常态，800 行上限；函数 < 50 行。

## 本轮新增的三条实测修正（写计划期间发现，设计文档需回填）

1. **库的 `real_time_factor` 与文档定义互为倒数。** 源码 `rtf = audio_duration / elapsed`（>1 为快于实时），文档定义 RTF = 耗时/音频时长（<1 为快）。**manifest 只记原始的 `generation_seconds` 与 `audio_seconds`**，RTF 由消费方派生，避免语义反转。
2. **截断上限是可算的，不是未知的。** 生成循环跑 `max_tokens` 个 patch，每 patch = `patch_size(4) × decode_chunk_size(1920) / 48000` = **0.16 秒音频**。默认 `max_tokens=2000` → **320 秒上限** → 按实测 0.2127 秒/汉字约 **1504 汉字**。所以本轮显式传 `max_tokens`（由字数推算 + 余量），把截断从「事后由闸门 1 发现」变成「事前由预算断言排除」。设计 §11「真实上限未知」由此关闭。
3. **参考音频在模型内部被降采样到 16 kHz**（`_encode_sample_rate=16000`），且传路径时库会自行重采样。故设计 §9「锚点非 48kHz 就拒绝」的原因需改写：模型不在乎，**我们自己的度量在乎**——因此 `metrics` 一律显式接收 `sr`，不硬编码 48000；`profile` 仍要求 48kHz 锚点，理由是与 48kHz 输出做 LTAS 比较时保持带宽一致。

## 第四条修正：闸门 1 必须双侧归一化（否则首篇真实短帖会误判）

设计 §6 已指出「归一化后文本变纯中文，而 Whisper 仍把听到的中文数字写回阿拉伯数字」，但只推出「difflib 稳健性仍必需」。这不够：

- 合成的是 `第一百二十二天`（6 个汉字）；Whisper 写回 `第122天`，`han_only` 后只剩 `第天`。
- 真实短帖 2 含 `122` 三处 + `7 月 24 号` 等，归一化后约 15+ 个汉字在 ASR 侧无对应物，572 汉字里丢 2.6% → 覆盖率跌到 ~96%，**在完好无损的音频上触发闸门 1 失败**。

**解法：对 ASR 文本施加同一个归一化函数再比。** 数字形式差异被对称消掉，比较从「对数字盲」变成「对数字形式不变」。`textmatch` 保持纯 stdlib，归一化策略放在 `verify`。

---

## 文件结构

```
voxout/
  pyproject.toml                  uv 项目 + pytest 配置 + 清华镜像 + prerelease=allow
  src/voxout/__init__.py          设 HF_HUB_OFFLINE=1
  src/voxout/config.py            全部常量，每个阈值注明来源
  src/voxout/textmatch.py         han_only / coverage / missing_spans     纯 stdlib
  src/voxout/markdown.py          md → 可朗读文本                        纯
  src/voxout/normalize.py         年份/千分位/区间/单位 + cn2an          cn2an
  src/voxout/chunker.py           段落原子；>60 汉字的段落才切            纯
  src/voxout/metrics.py           F0 / p10 / LTAS / cosine               numpy
  src/voxout/budget.py            patch 预算 → max_tokens                纯
  src/voxout/profile.py           音色档案：铸造 / 加载 / 校验
  src/voxout/synth.py             持一个已加载模型，逐段生成
  src/voxout/verify.py            两道闸门
  src/voxout/assemble.py          拼接 + wav + manifest
  src/voxout/cli.py               mint-voice / render
  tests/unit/test_*.py            不碰模型
  tests/integration/test_*.py     碰模型，标记 integration
  tests/fixtures/probe_*.py       本次探测脚本升格为回归装置
```

---

### Task 1: 项目骨架与常量

**Files:** Create `pyproject.toml`, `src/voxout/__init__.py`, `src/voxout/config.py`, `tests/unit/test_config.py`

- [ ] pyproject：`requires-python=">=3.12"`，deps `mlx-audio[tts,stt]>=0.4.8`、`cn2an>=0.5.24`、`numpy>=2.0`、`scipy>=1.18`；dev deps `pytest`；`[tool.uv] prerelease="allow"`（transformers 5.15.0 需要）；`[[tool.uv.index]]` 指向 `https://pypi.tuna.tsinghua.edu.cn/simple` 且 `default=true`（设计 §4 坑 1：默认源静默挂死 29 分钟）；`[tool.pytest.ini_options]` 注册 `integration` marker，`pythonpath=["src"]`，不设 addopts（显式优于隐式）。
- [ ] `__init__.py`：`os.environ.setdefault("HF_HUB_OFFLINE", "1")` + 一行注释指向设计 §9。
- [ ] `config.py`：常量见 Global Constraints，每条注明来源（实测值/设计章节）。
- [ ] 测试：阈值自洽性——`0 < COVERAGE_MIN < 1`；`CHUNK_THRESHOLD_HAN * SECONDS_PER_HAN_ESTIMATE < MAX_TOKENS_CAP * 0.16`（切段阈值必须落在 token 预算内，否则两个机制互相矛盾）；`F0_MIN_HZ < F0_MAX_HZ`；`VOICE_ROOT` 是绝对路径。
- [ ] `uv sync` 成功；`uv run pytest tests/unit -q` 通过。
- [ ] Commit `chore: voxout 项目骨架与常量`

### Task 2: textmatch.py（含贪心失步 bug 的回归锁）

**Interfaces:** `han_only(text: str) -> str`；`coverage(reference: str, heard: str) -> float`；`missing_spans(reference: str, heard: str) -> tuple[tuple[int, str], ...]`（返回 (起始下标, 缺失片段)）

- [ ] 测试先行：
  - 完全一致 → 1.0
  - 尾部缺失：`"一二三四五六七八九十"` vs `"一二三四五"` → 0.5
  - **回归锁（设计 §6）**：`coverage("第一百二十二天读了三页", "第一百二十二天读了3页")` ≥ 0.9 —— 局部替换不得引起永久失步。手写贪心版会在此崩到 3.5%。
  - 空参考 → 0.0
  - `missing_spans("一二三四五六", "一二五六")` → `((2, "三四"),)`
- [ ] 实现：`difflib.SequenceMatcher(None, ref, got, autojunk=False)`，覆盖率 = 匹配块 size 之和 / len(ref)；`missing_spans` 用 `get_opcodes()` 取 `delete`/`replace` 段。
- [ ] Commit `feat: textmatch 汉字覆盖率与缺失区段`

### Task 3: markdown.py

**Interfaces:** `to_speakable(md: str) -> str`

按真实语料（两篇短帖）确定需处理的构件，不做投机泛化：

- [ ] 测试：YAML frontmatter 整块剥离；`# 标题` → 保留标题文本成独立句；`[[wiki|别名]]` → 别名，`[[wiki]]` → wiki；`[文本](url)` → 文本；```围栏代码块``` → 整块丢弃；`` `行内代码` `` → 保留内容；`**粗** *斜*` → 去标记；`> 引用` → 去标记；`- 列表` → 去标记；`![图](x)` → 丢弃；`<!-- 注释 -->` → 丢弃；空输入 → 抛 `ValueError`；清洗后无内容 → 抛 `ValueError`（设计 §9：不产出静音文件）。
- [ ] 实现：纯正则流水线，段落间用 `\n` 保留（chunker 依赖它切段）。
- [ ] Commit `feat: markdown 清洗为可朗读文本`

### Task 4: normalize.py（消除误读风险，而非检测）

**Interfaces:** `to_speakable_numbers(text: str) -> str`；`UNIT_READINGS: dict[str, str]`

处理顺序（实测 cn2an 的失败模式决定，见下表）：千分位 → 年份 → 区间 → 单位 → cn2an → 折叠空白。

| 输入 | cn2an 裸输出 | 正确 | 本模块前置规则 |
|---|---|---|---|
| `2026 年` | `二千零二十六 年` ❌ | `二零二六年` | 4 位数字 + `年` → 逐位读 |
| `1,200 元` | `一,二百 元` ❌ | `一千二百元` | 数字组内逗号先删 |
| `第 3-5 章` | `第 三负五 章` ❌ | `第三到五章` | `\d+[-–~]\d+` → `X 到 Y` |
| `-5%` | `百分之负五` ❌ | `负百分之五` | 前导负号提到 `百分之` 前 |
| `48kHz` | `四十八kHz` ❌ | `四十八千赫兹` | 单位字典 |
| `第 122 天` | `第 一百二十二 天` ✓ | 同 | 仅折叠空白 |
| `7 月 24 号` | `七 月 二十四 号` ✓ | 同 | 仅折叠空白 |
| `72.2%` | `百分之七十二点二` ✓ | 同 | — |

- [ ] `UNIT_READINGS`：`kHz→千赫兹`、`Hz→赫兹`、`GB→吉字节`、`MB→兆字节`、`KB→千字节`、`TB→太字节`、`ms→毫秒`、`km→公里`、`cm→厘米`、`kg→公斤`、`°C→摄氏度`。**用户可编辑，且需人工听审确认自然度**（设计 §11）。按 key 长度降序替换，避免 `Hz` 先吃掉 `kHz`。
- [ ] 测试：上表每行一个用例（含 ✓ 行，锁住现状）；`cn2an` 抛异常时回退原文且返回值可标记（`to_speakable_numbers` 内部 try/except，失败时返回原文并 `logging.warning`，绝不静默）。
- [ ] Commit `feat: normalize 数字与单位归一化`

### Task 5: chunker.py

**Interfaces:** `chunk(text: str, threshold_han: int = CHUNK_THRESHOLD_HAN) -> tuple[str, ...]`

- [ ] 测试：单段落 60 汉字整好 → 不切（`>` 而非 `>=`）；61 汉字 → 多段；多段落输入 → 段落各自成段（原子化，T13a）；按 `。！？；\n` 切；无标点的超长句 → 不得丢字（返回单段并 `logging.warning`，切段不是丢内容的许可）；每段拼回去汉字数守恒（断言 `sum(han) == han(total)`）。
- [ ] Commit `feat: chunker 长文切段`

### Task 6: metrics.py（对闭式真值校准）

**Interfaces:** `frames_of(signal, frame=FRAME, hop=HOP)`；`voiced_f0s(signal, sr)`；`p10_f0(signal, sr)`；`median_f0(signal, sr)`；`ltas(signal, sr)`；`cosine(a, b)`；`spread_pct(values)`

代码由设计阶段的 `tests/probe_control.py` 与 `probe_voice_consistency.py` 升格（两者已删除，见设计 §10），**`sr` 改为显式参数**（修正 3）。

- [ ] 测试（全部有闭式真值，符合 lessons.md #1）：
  - 200 Hz 正弦 @48k，2s → `p10_f0` 与 `median_f0` 均在 200 ± 1%
  - **八度错误锁**：440 Hz 正弦 → 判为 220.2 Hz（断言 215 < est < 226）。原因：440 > F0_MAX_HZ=400，真周期 109 样本 < lag_min=120，峰值落在 2 倍周期。这是设计取值的后果，断言写死以防被误「修」
  - p10 vs 中位数的判别力：200 Hz 段 + 250 Hz 段拼接 → `p10_f0` ≈ 200，`median_f0` > 210（这正是设计 §4 那个方法论陷阱的量化形式）
  - `ltas` 自身余弦 = 1.0；200 Hz vs 2000 Hz 正弦余弦 < 0.5
  - 全静音输入 → `p10_f0` 返回 `nan` 而非崩溃
  - `spread_pct([100,100,100])` == 0
- [ ] Commit `feat: metrics F0/LTAS 原语（sr 显式化）`

### Task 7: budget.py（把截断变成事前断言）

**Interfaces:**
```python
def seconds_per_patch(patch_size: int, decode_chunk_size: int, sample_rate: int) -> float
def plan_max_tokens(n_han: int, sec_per_patch: float) -> int   # 抛 BudgetError 若超 MAX_TOKENS_CAP
class BudgetError(ValueError)
```

- [ ] 测试：`seconds_per_patch(4, 1920, 48000) == 0.16`；`plan_max_tokens(572, 0.16)` 落在 (1000, 1200] 区间（572 × 0.30 / 0.16 = 1072.5 → 1073 + 64 余量 = 1137）；`plan_max_tokens(1, 0.16)` ≥ 65（余量保底）；`plan_max_tokens(10_000, 0.16)` 抛 `BudgetError` 且消息含建议切段字数。
- [ ] Commit `feat: budget 事前 token 预算`

### Task 8: profile.py（音色档案）

**Interfaces:**
```python
@dataclass(frozen=True)
class VoiceProfile:
    name: str; anchor_wav: Path; anchor_text: str
    baseline_p10_f0: float; source: str; seed: int | None
    mint_text: str | None; created: str
def load_profile(name: str, root: Path = VOICE_ROOT) -> VoiceProfile
def mint_profile(name, text, seed, synthesizer, root=VOICE_ROOT) -> VoiceProfile
```

- [ ] 单测（用合成的 48k 正弦 wav 作 fixture，不碰模型）：
  - 完整档案 → 正常加载，字段齐全
  - 缺 `anchor.wav` → `FileNotFoundError`，消息**含铸造命令**（设计 §9）
  - 44100 Hz 锚点 → `ValueError`，消息含 ffmpeg 重采样命令（理由见修正 3：模型不在乎，我们的 LTAS 比较在乎）
  - 立体声锚点 → `ValueError`
  - 空 `anchor.txt` → `ValueError`
  - `baseline_p10_f0` 缺失或 ≤ 0 → `ValueError`
- [ ] `profile.toml` 用 stdlib `tomllib` 读；写用手拼字符串（stdlib 无 toml writer，字段少）。
- [ ] Commit `feat: profile 音色档案加载与校验`

### Task 9: synth.py（唯一碰生成模型的模块）

**Interfaces:**
```python
@dataclass(frozen=True)
class Segment:
    audio: np.ndarray          # float32, 1-D
    sample_rate: int
    audio_seconds: float
    generation_seconds: float
    token_count: int
    max_tokens: int
    peak_memory_gb: float

class Synthesizer:
    def __init__(self, model_id: str = MODEL_ID) -> None      # load() 一次，1.2–2.0s
    @property
    def sample_rate(self) -> int
    @property
    def seconds_per_patch(self) -> float                       # 从 model 配置派生，不硬编码
    def generate(self, text: str, *, ref_audio: Path | None = None,
                 ref_text: str | None = None, seed: int | None = None,
                 max_tokens: int | None = None) -> Segment
```

- [ ] `generate` 内：`mx.random.seed(seed)` 若给定；`max_tokens` 缺省由 `budget.plan_max_tokens` 算；`list(model.generate(...))` 取唯一 yield（源码 line 296 单 yield，`segment_idx=0`）；`np.asarray(r.audio, dtype=np.float32)`；**不记录库的 `real_time_factor`**（修正 1），只记 `processing_time_seconds` 与推导的 `audio_seconds`。
- [ ] 集成测试（`integration` 标记）：短文本 `"这是一段测试。"` → `sample_rate == 48000`、`audio_seconds > 0.5`、`token_count > 0`；同 `(seed, text)` 两次 → **逐样本 bit 相同**（`np.abs(a-b).max() == 0.0`，锁住设计 §3 的性质，锚点可复现依赖它）。
- [ ] Commit `feat: synth 单模型实例生成`

### Task 10: verify.py（两道闸门）

**Interfaces:**
```python
@dataclass(frozen=True)
class ContentGate:
    coverage: float; threshold: float; passed: bool
    asr_text: str; missing: tuple[tuple[int, str], ...]; asr_normalize_failed: bool
@dataclass(frozen=True)
class VoiceGate:
    baseline_p10: float; segment_p10s: tuple[float, ...]
    drift_pct: float; spread_pct: float; threshold: float; passed: bool

def check_content(wav_path: Path, spoken_text: str, transcriber) -> ContentGate
def check_voice(segments: Sequence[np.ndarray], sr: int, baseline_p10: float) -> VoiceGate
def load_transcriber(model_id: str = ASR_MODEL_ID)
```

- [ ] 闸门 1 关键点：`transcriber.generate(str(wav_path), language="zh", temperature=0.0)` —— 显式定语言与温度，让闸门可复现（默认温度是回退元组，同音频可能给不同数字）。**两侧都过 `normalize.to_speakable_numbers` 再比**（第四条修正）；ASR 侧归一化失败时置 `asr_normalize_failed=True` 并仍用原文比较。
- [ ] 闸门 2 关键点：`drift_pct = 100 * |p10_i - baseline| / baseline` 取最大；多段时另算段间 `spread_pct`；单段也能判（对锚点基线比）。
- [ ] 单测（不碰模型，传入假 transcriber）：
  - 假 transcriber 回原文 → coverage 1.0，passed
  - 假 transcriber 回截断文本 → passed False，`missing` 非空
  - **数字形式不一致的真实场景**：spoken=`"第一百二十二天"`，ASR 回 `"第122天"` → coverage 1.0 且 passed（若无双侧归一化此处会 ~0.29 → 这条测试就是第四条修正的锁）
  - 闸门 2：200 Hz 正弦 vs baseline 200 → drift ≈ 0 passed；250 Hz vs baseline 200 → drift 25% failed
- [ ] Commit `feat: verify 内容完整性与音色一致性两道闸门`

### Task 11: assemble.py

**Interfaces:** `join(segments: Sequence[np.ndarray], sr: int, gap_seconds: float = CHUNK_GAP_SECONDS) -> np.ndarray`；`write_wav(path, audio, sr) -> None`；`build_manifest(...) -> dict`；`write_manifest(path, manifest) -> None`

- [ ] manifest 字段：`input`、`voice`、`text_sha256`、`chars`/`han_chars`、`segments`、`audio_seconds`、`generation_seconds`、`seconds_per_han`（设计 §6 的独立旁证，必须进常规输出）、`max_tokens`、`token_count`、`peak_memory_gb`、`gate_content`、`gate_voice`、`model_id`、`asr_model_id`、`versions{mlx_audio,mlx}`、`created`、`forced`。
- [ ] 测试：单段 join = 原数组（零拷贝语义无关，值相等）；两段 join 长度 = 两段 + gap；`write_wav` 后读回 sr 与长度一致，dtype int16；`build_manifest` 含全部键且 `json.dumps` 可序列化（无 numpy 标量泄漏——`float()`/`int()` 显式转换）；`seconds_per_han` 计算正确。
- [ ] Commit `feat: assemble 拼接与 manifest`

### Task 12: cli.py 与真实铸造

**Interfaces:** `voxout mint-voice <name> [--text ...] [--seed 42]`；`voxout render <md> --voice <name> [-o path] [--force]`

- [ ] `render` 流水线：读 md → `markdown.to_speakable` → `normalize.to_speakable_numbers` → `chunker.chunk` → `profile.load_profile` → `Synthesizer` → 逐段 `generate(ref_audio=anchor, ref_text=anchor_text, max_tokens=预算)` → `assemble.join` → 写临时 wav → `verify.check_content`（不过且无 `--force` 则删除临时文件、写 `.diff.txt`、退出码 1）→ `verify.check_voice` → 写 `out/<slug>.wav` + `out/<slug>.manifest.json`。
- [ ] 错误处理按设计 §9 六行逐一对应，消息含可复制的修复命令。
- [ ] 真实铸造 `house` 音色：`mint-voice house --seed 42`，锚点文本取设计 §4 探测用的 `这是一段用来测试音色是否稳定的中文句子。`（其 bit 级可复现性已实测）。写入 `68 Writing Assets/voice/house/`。
- [ ] Commit `feat: cli mint-voice 与 render`

### Task 13: 端到端 + 闸门标定 + 回填文档

- [ ] 对两篇真实短帖各跑一次 `render`，记录 coverage 与 drift 实际值。
- [ ] **标定**：闸门 1 的 97% 是在「numeral-free 语料 + Whisper 默认参数」下测的 98.4–98.9%。本轮改了两处（显式 `language="zh"`/`temperature=0.0`、双侧归一化），**阈值必须在真实语料上重新读数**。若实测低于 97% 且音频经人工确认完整，则把阈值与理由一并更新，不许「调阈值让它过」而不写原因（lessons.md #1）。
- [ ] 独立旁证核对：两篇的 `seconds_per_han` 应在同一量级（漂移 < 15%），与 coverage 互为独立证据。
- [ ] 探测脚本迁入 `tests/fixtures/`，改为从 `voxout.metrics` 导入，删除脚本内重复的 F0 实现（DRY），保留正/负控制标尺。
- [ ] 回填设计文档：四条修正（RTF 倒数、token 预算可算、参考音 16k、闸门 1 双侧归一化）；§11 删除「真实上限未知」。
- [ ] 人工听审待办写入 `docs/`：单位字典自然度、锚点音色是否可接受、段间 gap 是否合适。
- [ ] Commit `feat: 端到端渲染短帖 + 闸门标定 + 文档回填`

## Self-Review

**Spec coverage:** §1 范围→T12/T13；§2 语言→Global Constraints；§3 证据→T9 集成测试锁 bit 相同；§4 锚点强制→T8/T12（`ref_audio` 非可选）；§5 架构模块表→T1–T12 逐一对应（新增 `config.py`/`budget.py`，理由已注明）；§6 两道闸门→T10 + T13 标定；§7 克隆升级→T8 档案格式使代码零改动；§8 长文→显式不在本轮（T5 只做切段，不做 continuation）；§9 错误处理六行→T12；§10 测试策略→T2/T6 的闭式真值与回归锁；§11 未覆盖→T13 回填。

**Type consistency:** `to_speakable`（markdown）与 `to_speakable_numbers`（normalize）名称不同不冲突；`Segment.audio` 全程 `np.ndarray float32 1-D`；`p10_f0(signal, sr)` 在 T6/T8/T10 三处签名一致；`VoiceProfile.baseline_p10_f0` 与 `VoiceGate.baseline_p10` 为不同层的同一数值，转换点在 T12。

**新增偏离（已在正文注明理由）：** `config.py`（DRY，阈值被 verify/cli/tests 三处消费）、`budget.py`（把截断变事前断言）、闸门 1 双侧归一化（否则首篇真实短帖误判）、`metrics` 的 `sr` 显式化（消除 44.1k 锚点导致 8.8% F0 偏差的隐藏 bug）。
