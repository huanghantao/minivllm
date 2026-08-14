"""W8：minivllm ↔ 真实 vLLM v1 模块对照图。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_ORANGE, arrow, box, canvas, save


def mapping():
    pairs = [
        ("scheduler/scheduler.py", "vllm/v1/core/sched/scheduler.py", "调度器"),
        ("cache/block_pool.py", "vllm/v1/core/block_pool.py", "块池"),
        ("cache/block_table.py", "vllm/v1/worker/block_table.py", "块表"),
        ("cache/paged_cache.py", "vllm/v1/attention/backends/*", "分页注意力"),
        ("sampling/sampler.py", "vllm/v1/sample/*  + SamplingParams", "采样器"),
        ("engine/llm_engine.py", "vllm/v1/engine/llm_engine.py", "引擎"),
        ("engine/llm.py", "vllm/entrypoints/llm.py", "用户门面"),
        ("model/qwen3.py", "vllm/model_executor/models/qwen3.py", "模型实现"),
    ]
    fig, ax = canvas(w=11.5, h=6.6, xlim=(0, 15), ylim=(-0.9, 10))
    ax.text(3.0, 9.4, "我们的 minivllm", fontsize=14, ha="center", weight="bold",
            color=C_BLUE)
    ax.text(12.0, 9.4, "真实的 vLLM v1", fontsize=14, ha="center", weight="bold",
            color=C_ORANGE)

    y = 8.5
    for mini, real, label in pairs:
        box(ax, (0.4, y - 0.35), 5.2, 0.72, mini, fc="#eef4fb", ec=C_BLUE, fontsize=10)
        box(ax, (9.4, y - 0.35), 5.2, 0.72, real, fc="#fff3e6", ec=C_ORANGE, fontsize=10)
        arrow(ax, (5.7, y), (9.3, y), color=C_GRAY)
        ax.text(7.5, y + 0.22, label, fontsize=10, ha="center", color="#555555")
        y -= 1.06

    ax.text(7.5, -0.45, "左边每一行你都亲手写过——右边就是它在大人世界里的样子。",
            fontsize=12, ha="center", color="#555555")
    save(fig, "w8_module_map.png")


if __name__ == "__main__":
    mapping()
