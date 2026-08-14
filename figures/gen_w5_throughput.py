"""W5：实测——我们自己的引擎，1 个请求 vs 8 个请求的吞吐。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GREEN, load_data, save

import matplotlib.pyplot as plt


def main():
    d = load_data("bench.json")
    b1, b8 = d["minivllm_b1_tps"], d["minivllm_b8_tps"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(["一次只服务 1 个请求", "continuous batching\n8 个请求一起跑"],
                  [b1, b8], color=[C_BLUE, C_GREEN], width=0.5)
    for bar, v in zip(bars, [b1, b8]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f} tok/s",
                ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("总吞吐（token/秒）")
    ax.set_title(f"minivllm 实测（Qwen3-0.6B，MPS）：批处理把吞吐抬高 {b8 / b1:.1f} 倍")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "w5_throughput.png")


if __name__ == "__main__":
    main()
