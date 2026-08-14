"""W6：temperature 与 top-p 的真实效果（用我们自己的 sampler 算）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_ORANGE, C_RED, save

import matplotlib.pyplot as plt
import numpy as np
import torch


def temperature():
    from reference.sampling.sampler import apply_temperature

    words = ["的", "是", "我", "了", "在", "吃", "苹", "果"]
    torch.manual_seed(7)
    logits = torch.randn(8) * 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, T in zip(axes, [0.5, 1.0, 2.0]):
        probs = torch.softmax(apply_temperature(logits, T), dim=-1).numpy()
        ax.bar(words, probs, color=C_BLUE, alpha=0.85)
        ax.set_title(f"temperature = {T}", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("选中概率")
    fig.suptitle("同一组 logits，三种温度：T 越小越「认死理」，T 越大越「端水」",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "w6_temperature.png")


def top_p():
    from reference.sampling.sampler import top_p_filter

    torch.manual_seed(3)
    vocab = 12
    logits = torch.randn(vocab) * 1.5
    p = 0.8

    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order].numpy()
    cum = np.cumsum(sorted_probs)
    kept = top_p_filter(logits, p)
    kept_ids = set(torch.isfinite(kept).nonzero().flatten().tolist())
    kept_flags = [int(i) in kept_ids for i in order.tolist()]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = [C_BLUE if f else C_GRAY for f in kept_flags]
    x = np.arange(vocab)
    ax.bar(x, sorted_probs, color=colors, alpha=0.9, label="概率（按高低排序）")
    ax2 = ax.twinx()
    ax2.plot(x, cum, "o-", color=C_RED, label="累计概率")
    ax2.axhline(p, color=C_RED, ls="--", lw=1)
    ax2.text(vocab - 0.5, p + 0.02, f"p = {p}", color=C_RED, ha="right", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("累计概率", color=C_RED)
    ax.set_xlabel("token（按概率从高到低排）")
    ax.set_ylabel("概率")
    ax.set_title("top-p（核采样）：蓝色留下、灰色出局——累计概率刚盖过 p 就关门")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    save(fig, "w6_top_p.png")


if __name__ == "__main__":
    temperature()
    top_p()
