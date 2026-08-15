# voxout

把 markdown 内容渲染成音色固定的 48kHz 对外音频。

VoxCPM2 渲染 + 音色锚点 + 两道自动闸门。它是个 CLI 工具，没有服务端、端口或常驻进程。

## 渲染一篇

新机器先走一遍下面的〈换机器〉，`house` 音色铸好之后才有这一步。

```bash
cd ~/workspace/claude/voxout
uv run voxout render <某篇.md> --voice house
```

不加 `-o` 时输出到 `out/<帖子名>.wav`（中文保留，空白折成连字符）。

实测例子：

```console
$ uv run voxout render out/smoke.md --voice house -o out/smoke.wav
切成 3 段，共 56 字
第 1/3 段完成：1.4 秒音频，耗时 2.0 秒
第 2/3 段完成：5.9 秒音频，耗时 5.5 秒
第 3/3 段完成：4.3 秒音频，耗时 4.0 秒
✓ 渲染完成  out/smoke.wav
  内容闸门  覆盖率 100.00%（阈值 97%）  通过
  音色闸门  漂移 10.70%（上限 15%）  通过  段间极差 46.90%
  时长 12.1 秒 / 49 汉字  0.2384 秒每汉字  生成耗时 11.5 秒
  manifest  out/smoke.manifest.json
```

产物是 48kHz / 单声道 / 16-bit PCM，旁边一份同名 `.manifest.json` 记下这次渲染的
全部读数（覆盖率、漂移、耗时、token、峰值内存、模型与库版本）。

## 两道闸门

渲染完自动跑，判据与阈值的完整标定过程在 `src/voxout/config.py` 的常量文档里。

**闸门 1（内容完整性）**：把产出音频用 whisper 回环转录，比对汉字覆盖率，阈值 **97%**。
不过且未 `--force` 时**不产出音频**——一个丢了内容的音频比没有文件更危险，它看起来
像成品。只在不过时写一份差异报告，路径是输出 wav 换后缀（`out/smoke.wav` →
`out/smoke.diff.txt`）：带位置的缺失片段、归一化后**模型实际收到的**朗读文本、
以及 ASR 原话。三样并排，才能判断是 ASR 听错还是真丢了内容——这是决定要不要
`--force` 的唯一依据。

**闸门 2（音色一致性）**：各段 p10 F0 相对锚点基线的漂移，取绝对值的**中位数**，
上限 **15%**。不过只告警不拦——p10 F0 是代理指标，会误报，该由人试听决定。

> 短标题被切成独立段落时，段间极差会很大（上例的 46.90% 来自 1.4 秒的标题段，
> 三段 p10 F0 是 `[231.4, 153.8, 148.1]`）。这么短的音频上 p10 F0 基本是噪声，
> 所以判据取中位数，不取极差。

## render 的参数

| 参数 | 作用 |
|---|---|
| `--voice NAME` | 必填。音色名，例如 `house` |
| `-o, --output PATH` | 输出 wav，默认 `out/<帖子名>.wav` |
| `--force` | 内容闸门不过也产出音频，manifest 记 `forced=true` |
| `--voice-root PATH` | 音色档案根目录，默认见〈音色档案〉 |

## 退出码

退出码是它能进脚本的契约。

| 码 | 含义 |
|---|---|
| `0` | 成功。音色闸门可能只是告警——它的判据是代理指标，不挡流程 |
| `1` | 内容闸门不过。**预期内**的失败，看过差异报告后重跑或 `--force` |
| `2` | 操作性错误：档案缺失、输入为空、权重没缓存。argparse 用法错误同码 |

## 音色档案

`house` 已经铸好，日常不用管这一层。档案默认在：

```
    anchor.wav      锚点音频（48kHz）
    anchor.txt      锚点文本
    profile.toml    基线 p10 F0、种子、铸造时间
```

退出码 2 的「档案缺失」指的就是这个目录少了文件，最常见的起因是 `--voice` 拼错。

## 换机器

已配好的机器不需要任何安装步骤。换机器时按顺序走三步。

**1. 装依赖**

```bash
uv sync
```

需要 Python ≥ 3.12 与 [uv](https://docs.astral.sh/uv/)。依赖走清华镜像（`files.pythonhosted.org`
在本机网络下会静默挂死，实测 29 分钟零进展，见 `pyproject.toml`）。

**2. 拉模型权重**

包默认离线（`HF_HUB_OFFLINE=1`，设在 `__init__.py`，必须早于任何 `huggingface_hub`
导入），所以新机器第一次运行必然失败一次，并打印可直接粘贴的下载命令：

| 模型 | 用途 | 大小 |
|---|---|---|
| `mlx-community/VoxCPM2-8bit` | TTS 渲染 | 3.0GB |
| `mlx-community/whisper-large-v3-turbo-asr-fp16` | 闸门 1 回环转录 | 1.5GB |

两个都要拉，注意 TTS 走 `tts.utils`、ASR 走 `stt.utils`：

```bash
export HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1

uv run python -c "from mlx_audio.tts.utils import load; \
  load('mlx-community/VoxCPM2-8bit')"
uv run python -c "from mlx_audio.stt.utils import load; \
  load('mlx-community/whisper-large-v3-turbo-asr-fp16')"
```

**3. 铸音色**

```bash
uv run voxout mint-voice house --seed 42
```

一个音色一生只铸一次，此后所有渲染都对齐它。`--text` 可换锚点文本，但默认那句是
设计 §4 全部探测的标定基准——换掉就要重新标定闸门 2 的标尺。`--seed` 不给则这次
铸造不可复现。铸完就能渲染了。

## 开发

```bash
uv run pytest -m 'not integration'   # 快速回路，242 项 / 0.7 秒，不碰模型
uv run pytest                        # 含 integration，会加载 3.0GB 权重，慢
uv run ruff check                    # 只用 check，不用 format，理由见 pyproject.toml
```

两个昂贵资源（TTS 与 ASR 权重）在 CLI 层由工厂注入，单测传替身，因此完全不碰模型。

## 文档

- `docs/superpowers/specs/2026-08-13-voxcpm2-content-tts-design.md` —— 设计与选型
- `docs/superpowers/plans/2026-08-14-voxout-implementation.md` —— 实施计划
- `docs/listening-review.md` —— 人工听审记录
- `src/voxout/config.py` —— 每个阈值的测量来源
