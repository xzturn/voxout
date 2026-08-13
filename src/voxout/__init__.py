"""voxout —— 把 markdown 内容渲染成音色固定的 48kHz 对外音频。

离线优先（设计 §9）：本机网络访问 huggingface.co 会静默挂死而非报错，实测挂 29 分钟
零进展。权重已缓存 3.0GB，宁可离线时明确失败，也不要静默挂死。必须在任何
mlx_audio / huggingface_hub 导入之前生效，故置于包的 __init__。
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

__all__ = ["__version__"]
__version__ = "0.1.0"
