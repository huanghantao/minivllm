"""W4：内存碎片 vs 分页（储物柜类比）+ 块表映射（两张）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_GREEN, C_ORANGE, C_RED, arrow, canvas, save

import matplotlib.pyplot as plt


def fragmentation():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：连续分配——长请求要一大块连着的空间，走后留下塞不下的洞
    ax = axes[0]
    ax.set_title("连续分配：走后留下「洞」（碎片）", fontsize=12)
    # 内存条 16 格
    layout = [
        (0, 4, C_BLUE, "请求A\n(4)"),
        (4, 2, C_GRAY, "空洞(2)"),
        (6, 5, C_GREEN, "请求B\n(5)"),
        (11, 3, C_GRAY, "空洞(3)"),
        (14, 2, C_ORANGE, "请求C\n(2)"),
    ]
    for start, w, color, label in layout:
        ax.add_patch(plt.Rectangle((start, 0), w - 0.08, 1, facecolor=color, alpha=0.8))
        ax.text(start + w / 2, 0.5, label, ha="center", va="center", fontsize=9,
                color="white" if color != C_GRAY else "#333333")
    ax.text(8, -0.75, "空洞加起来有 5 格，\n但新请求要连着的 4 格 → 塞不下！",
            ha="center", fontsize=10, color=C_RED)
    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    # 右：分页——固定大小的储物柜，谁来都能塞
    ax = axes[1]
    ax.set_title("分页：固定大小的块，谁来都能塞", fontsize=12)
    blocks = [C_BLUE, C_BLUE, C_GREEN, C_BLUE, C_GRAY, C_GREEN, C_ORANGE, C_GREEN,
              C_BLUE, C_GREEN, C_GRAY, C_ORANGE, C_GREEN, C_GRAY, C_GRAY, C_GRAY]
    for i, color in enumerate(blocks):
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 1, facecolor=color, alpha=0.8))
        ax.text(i + 0.46, 0.5, str(i), ha="center", va="center", fontsize=8,
                color="white" if color != C_GRAY else "#333333")
    ax.text(8, -0.75, "请求A 占了 0、1、3、8 号柜——不连着也没关系，\n块表记得住（见右图/下一图）",
            ha="center", fontsize=10, color=C_BLUE)
    ax.set_xlim(-0.3, 16.3)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    save(fig, "w4_fragmentation.png")


def block_table():
    """逻辑 token → 物理槽位：块表就是页表。"""
    fig, ax = canvas(w=11, h=5.2, xlim=(0, 14), ylim=(0, 7))

    # 上排：请求眼里连续的 10 个 token（逻辑视图）
    ax.text(7, 6.35, "请求眼里：连续的 10 个 token（逻辑视图）", fontsize=12,
            ha="center")
    seg_colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for i in range(10):
        seg = min(i // 4, 2)
        x = 1.2 + i * 0.95
        ax.add_patch(plt.Rectangle((x, 5.2), 0.83, 0.8, facecolor=seg_colors[seg],
                                   alpha=0.9))
        ax.text(x + 0.41, 5.6, str(i), ha="center", va="center", fontsize=11,
                color="white", weight="bold")

    # 下排：物理 KV cache，8 个块 × 4 格（物理视图）
    ax.text(7, 3.15, "物理 KV cache（块大小 = 4，块可以随便挑）", fontsize=12,
            ha="center")
    block_owner = {3: 0, 7: 1, 1: 2}  # 物理块 -> 第几段逻辑 token
    for b in range(8):
        x0 = 1.05 + b * 1.5
        owner = block_owner.get(b)
        face = seg_colors[owner] if owner is not None else "#dddddd"
        for s in range(4):
            ax.add_patch(plt.Rectangle((x0 + s * 0.32, 1.9), 0.28, 0.8,
                                       facecolor=face, alpha=0.9))
        ax.text(x0 + 0.62, 1.55, f"块{b}", ha="center", fontsize=10,
                color="#333333")

    # 映射箭头：逻辑段中心 → 物理块中心
    seg_targets = [(0, 3), (1, 7), (2, 1)]  # 第几段 -> 物理块
    for seg, b in seg_targets:
        x1 = 1.2 + (seg * 4 + 1.5) * 0.95 + 0.41  # 段中心
        x2 = 1.05 + b * 1.5 + 0.62               # 块中心
        rad = 0.22 if seg != 1 else -0.18
        arrow(ax, (x1, 5.1), (x2, 2.85), color=seg_colors[seg], lw=2.2,
              connectionstyle=f"arc3,rad={rad}")

    # 块表
    ax.text(12.7, 6.0, "块表（页表）", fontsize=12, ha="center")
    ax.text(12.7, 4.5, "段0 → 块3\n段1 → 块7\n段2 → 块1", fontsize=11, ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fb", ec=C_BLUE))

    ax.text(7, 0.55,
            "逻辑上连续的 token，物理上散在块 3、7、1 里；注意力计算前按块表 gather 回来，结果一模一样。",
            fontsize=11, ha="center", color="#555555")
    save(fig, "w4_block_table.png")


if __name__ == "__main__":
    fragmentation()
    block_table()
