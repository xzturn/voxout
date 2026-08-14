"""模型加载：把「离线缓存未命中」翻译成一条可直接粘贴的下载命令。

包默认离线（`__init__` 里设了 `HF_HUB_OFFLINE=1`），这是为了让渲染的行为
不依赖网络状况。代价是新机器上第一次运行必然失败一次，而 huggingface_hub
抛出的异常对使用者几乎没有信息量。

TTS 与 ASR 两处加载共用这段翻译——错误消息的结构完全相同，只有 import 路径
不同，没有理由写两遍。
"""

from collections.abc import Callable


def load_or_explain[Model](
    load: Callable[[str], Model], model_id: str, import_path: str
) -> Model:
    """加载模型；失败时换成带修复命令的 RuntimeError 抛出。

    Args:
        load: 库提供的加载函数，接受 model_id。
        model_id: HuggingFace 仓库 id。
        import_path: 修复命令里要写的模块路径，如 `mlx_audio.tts.utils`。

    Raises:
        RuntimeError: 加载失败。原异常保留在 `__cause__` 里——离线只是最常见的
            原因，不是唯一原因，真实异常必须能被追到。
    """
    try:
        return load(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"加载模型 {model_id} 失败：{exc}\n"
            f"默认离线（HF_HUB_OFFLINE=1）。若权重未缓存，先联网拉一次：\n"
            f"  HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com "
            f"HF_HUB_DISABLE_XET=1 uv run python -c "
            f"\"from {import_path} import load; load('{model_id}')\""
        ) from exc
