"""模型加载的失败路径。

这条路径以前散在 synth 里、且没有测试——而它恰恰是新机器上第一次运行时
唯一会碰到的错误。消息里必须有可直接粘贴的下载命令。
"""

import pytest

from voxout.hub import load_or_explain


def test_successful_load_is_passed_through_unchanged() -> None:
    loaded = load_or_explain(lambda model_id: f"loaded:{model_id}", "org/model", "pkg.utils")

    assert loaded == "loaded:org/model"


def test_failure_names_the_model_the_offline_flag_and_a_fix_command() -> None:
    def boom(_model_id: str) -> str:
        raise OSError("缓存未命中")

    with pytest.raises(RuntimeError) as excinfo:
        load_or_explain(boom, "org/model", "pkg.utils")

    message = str(excinfo.value)
    assert "org/model" in message
    assert "HF_HUB_OFFLINE" in message
    assert "from pkg.utils import load" in message


def test_original_exception_is_kept_as_the_cause() -> None:
    """离线只是最常见的原因，不是唯一原因。真实异常必须能被追到。"""
    original = OSError("磁盘损坏")

    def boom(_model_id: str) -> str:
        raise original

    with pytest.raises(RuntimeError) as excinfo:
        load_or_explain(boom, "org/model", "pkg.utils")

    assert excinfo.value.__cause__ is original
