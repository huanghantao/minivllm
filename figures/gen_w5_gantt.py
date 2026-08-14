"""W5：静态批处理 vs continuous batching 的甘特图（vLLM 论文里最经典的那张）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_GREEN, C_ORANGE, save

import matplotlib.pyplot as plt


def gantt():
    # 三个请求：A 生成 3 个词，B 生成 7 个，C 生成 4 个
    reqs = [("A", 3, C_BLUE), ("B", 7, C_ORANGE), ("C", 4, C_GREEN)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)

    # 上：静态批处理——一锅出，短的陪跑
    ax = axes[0]
    max_len = max(n for _, n, _ in reqs)
    for row, (name, n, color) in enumerate(reqs):
        ax.barh(row, n, left=0, height=0.62, color=color, alpha=0.9)
        ax.barh(row, max_len - n, left=n, height=0.62, color=C_GRAY, alpha=0.45,
                hatch="//", edgecolor="#666666")
        ax.text(n / 2, row, f"请求{name}：生成 {n} 词", ha="center", va="center",
                fontsize=10, color="white", weight="bold")
        if max_len - n > 0:
            ax.text(n + (max_len - n) / 2, row, "陪跑", ha="center", va="center",
                    fontsize=9, color="#444444")
    ax.set_yticks([0, 1, 2], ["", "", ""])
    ax.set_title("静态批处理：一锅同时出锅，短请求陪跑（斜线 = GPU 空转）", fontsize=12)
    ax.set_ylabel("批次", fontsize=10)

    # 下：continuous batching——谁做完谁下车，空位立刻上新人
    ax = axes[1]
    timeline = [(0, "A", 3, C_BLUE), (0, "B", 7, C_ORANGE)]
    # A 在第 3 步下车，C 上车；B 第 7 步下车
    ax.barh(1, 3, left=0, height=0.62, color=C_BLUE, alpha=0.9)
    ax.text(1.5, 1, "A：3 词", ha="center", va="center", fontsize=10,
            color="white", weight="bold")
    ax.barh(1, 4, left=3, height=0.62, color=C_GREEN, alpha=0.9)
    ax.text(5, 1, "C：4 词（A 下车，C 立刻上车）", ha="center", va="center",
            fontsize=10, color="white", weight="bold")
    ax.barh(0, 7, left=0, height=0.62, color=C_ORANGE, alpha=0.9)
    ax.text(3.5, 0, "B：7 词", ha="center", va="center", fontsize=10,
            color="white", weight="bold")
    ax.set_title("continuous batching：谁做完谁下车，空位立刻补人——GPU 不空转",
                 fontsize=12)
    ax.set_xlabel("时间（步）", fontsize=10)
    ax.set_ylabel("批次", fontsize=10)
    ax.set_yticks([])

    for ax in axes:
        ax.set_xlim(0, 7.5)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "w5_gantt.png")


if __name__ == "__main__":
    gantt()
