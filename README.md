# voxout

本机 CLI，不是服务。VoxCPM2 出声，音色锚点锁住说话人，两道闸门拦明显翻车。渲染和转录都走 MLX，要一台 Apple Silicon 的 Mac，没有 CPU 或 CUDA 退路。

## 跑起来

音色没铸好之前，先把文末的安装走完。日常两条命令：

```bash
uv run voxout mint-voice house --seed 42   # 一个音色只铸一次
uv run voxout render 某篇.md --voice house
```

不加 `-o` 时写到 `out/<帖子名>.wav`。中文保留，空白折成连字符。

实测一次长这样：

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

成品是 48kHz / 单声道 / 16-bit PCM。旁边一份同名 `.manifest.json`，记下覆盖率、漂移、耗时、token、峰值内存，以及模型与库版本。

`render` 另外几个参数：

| 参数 | 作用 |
|---|---|
| `--voice NAME` | 必填。音色名，例如 `house` |
| `-o, --output PATH` | 输出 wav，默认 `out/<帖子名>.wav` |
| `--force` | 内容闸门不过也产出音频，manifest 记 `forced=true` |
| `--voice-root PATH` | 音色档案根目录，默认 `~/.voxout/voice` |

## 渲染怎么走

markdown 先洗成朗读文本，按段落切开，每段带着锚点合成，拼接，过闸门，写成 wav。

段落是原子单位。跨段一次生成会跳读，这种错只有闸门 1 看得见。切得太碎也不好：单独一句短标题上，p10 F0 基本是噪声，所以音色判据用中位数，不用极差。上面那次段间极差 46.90%，来自 1.4 秒的标题段，三段 p10 F0 是 `[231.4, 153.8, 148.1]`。

## 两道闸门

渲染完自动跑。阈值怎么标定的，写在 `src/voxout/config.py` 的常量注释里，这里不复述。

**内容闸门**：whisper 回环，汉字覆盖率下限 97%（`COVERAGE_MIN`）。不过、又没加 `--force` 时，不写成品 wav，只写 `.diff.txt`。丢了内容的音频比没有文件更糟，它看起来像成品。报告跟输出 wav 换后缀（`out/smoke.wav` → `out/smoke.diff.txt`），三样并排：带位置的缺失片段、归一化后模型实际收到的朗读文本、ASR 原话。要不要 `--force`，看这一份。

**音色闸门**：各段 p10 F0 相对锚点基线的漂移，取绝对值的中位数，上限 15%（`VOICE_DRIFT_MAX_PCT`）。不过只告警，音频照出。这是代理指标，会误报，最后听耳朵。

## 退出码

退出码是它能进脚本的契约。

| 码 | 意思 |
|---|---|
| `0` | 成了。音色告警也算成——判据是代理指标，不挡流程 |
| `1` | 内容闸门不过。预期内的失败，看过差异报告再决定重跑还是 `--force` |
| `2` | 档案没了、输入为空、权重没缓存、参数写错。argparse 用法错误同码 |

档案缺失多半是 `--voice` 拼错。

## 音色档案

一个音色一份，铸好之后日常不用管。默认在：

```
~/.voxout/voice/<音色名>/
    anchor.wav      锚点音频（48kHz）
    anchor.txt      锚点文本
    profile.toml    基线 p10 F0、种子、铸造时间
```

档案是长期资产：一个音色一生只铸一次，之后所有渲染都对齐它。所以放在 home 下，不进仓库。换仓库、删仓库、重新 clone，已铸好的音色还在。

根目录有三层覆盖，越靠外越优先。`--voice-root` 只管这一次。`VOXOUT_VOICE_ROOT` 写进 shell 配置，档案放在别处（比如随笔记库一起同步）时用。都没设就用 `~/.voxout/voice`。

## 安装

按顺序三步。这台机器走完就不用再管，换机器再走一遍。

**1. 装依赖**

```bash
uv sync
```

需要 Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)，以及一台 Apple Silicon 的 Mac。

`pyproject.toml` 故意不钉索引镜像。索引是每台机器的网络事实，不该由项目强加。某些网络下默认的 `files.pythonhosted.org` 会静默挂死（实测 29 分钟零进展），这时在本机换一个：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple   # 按所在网络择一
uv sync
```

**2. 拉模型权重**

包默认离线（`HF_HUB_OFFLINE=1`，设在 `__init__.py`，必须早于任何 `huggingface_hub` 导入）。新机器第一次运行会失败，并打印一条可直接粘贴的下载命令。

两个都要拉，合计约 4.5GB。TTS 走 `tts.utils`，ASR 走 `stt.utils`：

- `mlx-community/VoxCPM2-8bit`，渲染，3.0GB
- `mlx-community/whisper-large-v3-turbo-asr-fp16`，闸门 1 回环转录，1.5GB

```bash
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1
# 直连拉不动时再加一个镜像端点，例如：
#   export HF_ENDPOINT=https://hf-mirror.com

uv run python -c "from mlx_audio.tts.utils import load; \
  load('mlx-community/VoxCPM2-8bit')"
uv run python -c "from mlx_audio.stt.utils import load; \
  load('mlx-community/whisper-large-v3-turbo-asr-fp16')"
```

失败信息里打出来的那条命令会带上 `HF_ENDPOINT=https://hf-mirror.com`。直连能通，就不必设镜像。

**3. 铸音色**

```bash
uv run voxout mint-voice house --seed 42
```

`house` 只是个名字，随便取，本文档的例子都用它。`--text` 可以换锚点文本，但默认那句是设计 §4 全部探测的标定基准，换掉就要重新标定闸门 2。`--seed` 不给，这次铸造不可复现。铸完就能渲染。

档案默认写到 `~/.voxout/voice/house/`。

## 开发

```bash
uv run pytest -m 'not integration'   # 快速回路，247 项 / 0.8 秒，不碰模型
uv run pytest                        # 含 integration，会加载 3.0GB 权重，慢
uv run ruff check                    # 只用 check，不用 format
```

`ruff` 不跑 `format`。它按码点算行宽，中文一个字占两列，现有代码是按肉眼宽度折的，formatter 会把中文重的行接回去。理由写在 `pyproject.toml`。

TTS 和 ASR 两套权重在 CLI 层由工厂注入，单测传替身，所以快速回路不碰模型。

代码大致这样摆：

```
src/voxout
  cli.py | 入口
  render.py | 整条流水线
  markdown.py / chunker.py | 文本准备
  synth.py / assemble.py | 出声与拼接
  verify.py | 两道闸门
  profile.py | 铸音色
  metrics.py / normalize.py / textmatch.py | 度量与比对
  config.py | 阈值来源
tests/unit | 快速回路，不碰模型
tests/integration | 真加载权重
docs/superpowers | 设计与实施计划
docs/listening-review.md | 人工听审
```

## 文档

- `docs/superpowers/specs/2026-08-13-voxcpm2-content-tts-design.md` —— 设计与选型
- `docs/superpowers/plans/2026-08-14-voxout-implementation.md` —— 实施计划
- `docs/listening-review.md` —— 人工听审
- `src/voxout/config.py` —— 每个阈值的测量来源
