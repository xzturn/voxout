"""包级别的唯一保证：导入 voxout 就等于进入离线模式。

设计 §9：本机访问 huggingface.co 会静默挂死而非报错（实测挂 29 分钟零进展）。
权重已缓存 3.0GB，所以宁可离线时明确失败。

这个默认值放在 `__init__` 而不是 CLI，是因为它必须早于任何 huggingface_hub
导入——而那可能发生在任何一次 `import voxout.x`（探针脚本、notebook、别的
调用方），不只是走 CLI 的时候。所以这里测的是「导入即生效」，不是某个函数。
"""

import importlib
import os

import pytest

import voxout


def test_importing_the_package_puts_it_in_offline_mode() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_an_explicit_setting_survives_the_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """新机器上第一次拉权重必须能联网。用 setdefault 而非赋值，就是为了留这条路。"""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    importlib.reload(voxout)

    assert os.environ["HF_HUB_OFFLINE"] == "0"
