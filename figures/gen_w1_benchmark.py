"""W1：实测对比——naive 写法 vs 真实 vLLM（见证终点）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_ORANGE, load_data, save

import matplotlib.pyplot as plt


def main():
    d = load_data("bench.json")
    naive = d["naive_hf_tps"]
    vllm = d["vllm_b1_tps"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(
        ["Week 1 的 naive 写法\n（每步重算整段）", "真实 vLLM\n（本课终点要对标的引擎）"],
        [naive, vllm], color=[C_GRAY, C_ORANGE], width=0.5,
    )
    for bar, v in zip(bars, [naive, vllm]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} tok/s",
                ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("生成速度（token/秒，越高越好）")
    ax.set_title(f"同一台 Mac、同一个 Qwen3-0.6B：vLLM 快 {vllm / naive:.0f} 倍")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "w1_benchmark.png")


if __name__ == "__main__":
    main()
