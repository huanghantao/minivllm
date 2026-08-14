"""测试的 IMPL 分发器。

测试默认打读者战场包 minivllm；设 IMPL=reference 则打参考答案：
    pytest tests                  # 读者填写进度（填之前是红的）
    IMPL=reference pytest tests   # 参考答案（必须全绿）
"""

import importlib
import os

IMPL = os.environ.get("IMPL", "minivllm")


def load(mod):
    """load("model.transformer") -> minivllm.model.transformer 或 reference 版。"""
    return importlib.import_module(f"{IMPL}.{mod}")
