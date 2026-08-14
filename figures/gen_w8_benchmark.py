"""W8：结业实测——minivllm vs 真实 vLLM。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_ORANGE, load_data, save

import matplotlib.pyplot as plt
import numpy as np


def main():
    d = load_data("bench.json")
    groups = ["1 个请求", "8 个请求"]
    mini = [d["minivllm_b1_tps"], d["minivllm_b8_tps"]]
    real = [d["vllm_b1_tps"], d["vllm_b8_tps"]]

    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, mini, w, label="我们的 minivllm", color=C_BLUE)
    b2 = ax.bar(x + w / 2, real, w, label="真实 vLLM", color=C_ORANGE)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, groups)
    ax.set_ylabel("吞吐（token/秒）")
    ax.set_title("结业对照：同一台 Mac、同一个模型——我们追到了真实引擎的几成功力？")
    ax.legend()
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "w8_benchmark.png")


if __name__ == "__main__":
    main()
