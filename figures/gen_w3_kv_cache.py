"""W3：naive 生成的重复计算浪费 + 实测加速曲线 + KV cache 内存账（三张）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_ORANGE, C_RED, save

import matplotlib.pyplot as plt
import numpy as np
import torch


def recompute_waste():
    """每一步，naive 都要把整段序列重算一遍。"""
    words = ["巴", "黎", "是", "法", "国", "首", "都"]
    steps = [3, 5, 7]  # 第 1/2/3 步时的序列长度
    fig, axes = plt.subplots(3, 1, figsize=(9, 3.6))
    for ax, L in zip(axes, steps):
        for i in range(L):
            color = C_ORANGE if i < L - 1 else C_RED
            label = "重复计算" if i < L - 1 else "真正新算"
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=color, alpha=0.75))
            if i < len(words):
                ax.text(i + 0.46, 0.5, words[i], ha="center", va="center",
                        fontsize=12, color="white", weight="bold")
        ax.set_xlim(-0.1, 7.2)
        ax.set_ylim(-0.35, 1.35)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f"第 {steps.index(L) + 1} 步", fontsize=11, rotation=0,
                      labelpad=28, va="center")
        for spine in ax.spines.values():
            spine.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(fc=C_ORANGE, alpha=0.75, label="重复计算（浪费）"),
                 Patch(fc=C_RED, alpha=0.75, label="真正新算（有用）")],
        loc="lower center", ncol=2, frameon=False, fontsize=11,
    )
    fig.suptitle("naive 生成：序列每变长一点，整段都要重算一遍", fontsize=13)
    fig.subplots_adjust(bottom=0.18, top=0.88, hspace=0.4)
    save(fig, "w3_recompute_waste.png")


def speedup_curve():
    """实测：同一模型，naive vs KV cache，每生成一个词的累计耗时。"""
    from reference.generate import generate_naive, generate_with_cache
    from reference.model.transformer import MiniConfig, MiniTransformer

    torch.manual_seed(0)
    model = MiniTransformer(MiniConfig(
        vocab_size=128, hidden_size=256, num_layers=4, num_heads=8, max_seq_len=96
    ))
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    n = 48

    import time

    def timed(fn):
        # 逐步计时：包一层 hook 太贵，改为测总时间再除；这里直接整段测
        start = time.perf_counter()
        fn(model, ids, max_new_tokens=n)
        return time.perf_counter() - start

    t_naive = timed(generate_naive)
    t_cached = timed(generate_with_cache)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(["naive\n（每步重算整段）", "KV cache\n（每步只算新词）"],
                  [t_naive, t_cached], color=[C_ORANGE, C_BLUE], width=0.5)
    for bar, t in zip(bars, [t_naive, t_cached]):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.2f} s",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylabel(f"生成 {n} 个词的总耗时（秒）")
    ax.set_title(f"KV cache 加速实测（4 层小模型，CPU）：快 {t_naive / t_cached:.1f} 倍")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "w3_speedup.png")
    print(f"  naive={t_naive:.2f}s cached={t_cached:.2f}s speedup={t_naive/t_cached:.1f}x")


def memory_bill():
    """按 Qwen3-0.6B 的真实尺寸，算 KV cache 内存账。"""
    from reference.cache.kv_cache import kv_cache_memory_bytes

    # Qwen3-0.6B：28 层，8 个 KV 头，每头 128 维，bf16
    layers, kv_heads, head_dim = 28, 8, 128
    seq_lens = np.array([128, 256, 512, 1024, 2048, 4096])
    mb_one = [kv_cache_memory_bytes(layers, kv_heads, head_dim, int(s)) / 2**20
              for s in seq_lens]
    mb_batch32 = [m * 32 for m in mb_one]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(seq_lens, mb_one, "o-", color=C_BLUE, label="1 个请求")
    ax.plot(seq_lens, mb_batch32, "s-", color=C_RED, label="32 个请求（一批）")
    for s, m in zip(seq_lens, mb_batch32):
        ax.annotate(f"{m / 1024:.1f} GB" if m >= 1024 else f"{m:.0f} MB",
                    (s, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=C_RED)
    ax.set_xlabel("序列长度（token 数）")
    ax.set_ylabel("KV cache 占用")
    ax.set_yscale("log")
    ax.set_yticks([100, 1000, 10000], ["100 MB", "1 GB", "10 GB"])
    ax.set_title("KV cache 内存账（Qwen3-0.6B，bf16）：每 token 约 112 KB")
    ax.legend()
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "w3_memory_bill.png")


if __name__ == "__main__":
    recompute_waste()
    speedup_curve()
    memory_bill()
