"""W2 概念桥：注意力输出如何变成 logits，以及训练/生成的两种 forward。"""

import pathlib
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import (  # noqa: E402
    C_BLUE,
    C_GRAY,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    arrow,
    canvas,
    save,
)


C_DARK = "#2f343b"
C_TEXT = "#4b5563"
C_LIGHT_BLUE = "#eef4fb"
C_LIGHT_GREEN = "#edf8ef"
C_LIGHT_ORANGE = "#fff5e8"
C_LIGHT_PURPLE = "#f5eff9"
C_LIGHT_RED = "#fff0f0"


def panel(ax, xy, w, h, title, fc="#fbfcfe", ec="#d7dde5"):
    """章节图使用的圆角面板。"""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.10",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.3,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.25,
        y + h - 0.28,
        title,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=C_DARK,
        zorder=5,
    )


def token_box(ax, xy, w, h, token, pos, fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, token, ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_DARK, zorder=4)
    ax.text(x + w / 2, y + 0.12, f"位置 {pos}", ha="center", va="bottom",
            fontsize=7.5, color="#6b7280", zorder=4)


def vector_box(ax, xy, w, h, label, values, fc=C_LIGHT_GREEN, ec=C_GREEN):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.69, label, ha="center", va="center",
            fontsize=9.6, fontweight="bold", color=C_DARK, zorder=4)
    text = "[" + ", ".join(f"{v:.1f}" for v in values) + "]"
    ax.text(x + w / 2, y + h * 0.34, text, ha="center", va="center",
            fontsize=8.5, color=C_TEXT, family="monospace", zorder=4)


def hidden_to_logits():
    """图 1：把 attention out、lm_head、logits 和 generate_naive 串成一条线。"""
    # 教学用 4 维隐藏向量与 5 词词表；logits 真由 h @ W.T 算出。
    words = ["我", "爱", "吃", "苹果", "EOS"]
    hidden = np.array([
        [0.2, 2.4, 0.1, -0.3],
        [0.1, 0.4, 3.2, 0.5],
    ])
    lm_head = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ])
    logits = hidden @ lm_head.T

    fig, ax = canvas(w=16, h=9.5, xlim=(0, 24), ylim=(0, 14.4))
    ax.text(12, 13.95, "注意力不生成新词：它只更新每个已有位置的向量",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#222222")
    ax.text(12, 13.34,
            "lm_head 才把每个位置的隐藏向量翻译成一整排词表 logits；generate_naive 最后只取最后一排",
            ha="center", va="center", fontsize=10.5, color="#555555")

    panel(ax, (0.45, 7.0), 4.0, 5.35, "① input_ids → Embedding", fc="#fbfdff", ec=C_BLUE)
    token_box(ax, (0.85, 9.75), 1.35, 1.15, "我", 0)
    token_box(ax, (2.65, 9.75), 1.35, 1.15, "爱", 1)
    ax.text(2.45, 9.25, "token id 不是可做注意力的内容\n先查表变成向量",
            ha="center", va="top", fontsize=9.2, color=C_TEXT)
    ax.text(2.45, 7.75, "形状  (1, 2) → (1, 2, E)",
            ha="center", va="center", fontsize=10, color=C_BLUE, fontweight="bold")

    panel(ax, (5.0, 7.0), 6.15, 5.35, "② Attention / Transformer blocks", fc="#fbfefb", ec=C_GREEN)
    vector_box(ax, (5.45, 9.75), 5.25, 1.15, "h₀：前缀 [我] 的上下文表示", hidden[0])
    vector_box(ax, (5.45, 8.05), 5.25, 1.15, "h₁：前缀 [我, 爱] 的上下文表示", hidden[1])
    ax.text(8.08, 7.50, "仍然只有 2 个位置；每个位置得到的是向量，不是词",
            ha="center", va="center", fontsize=9.6, color=C_GREEN, fontweight="bold")

    panel(ax, (11.7, 7.0), 2.9, 5.35, "③ lm_head", fc="#fffdf9", ec=C_ORANGE)
    ax.text(13.15, 10.55, "Linear", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_DARK)
    ax.text(13.15, 9.85, "E → vocab", ha="center", va="center",
            fontsize=10.5, color=C_ORANGE, fontweight="bold")
    ax.text(13.15, 8.85, "对 h₀、h₁\n分别做同一次\n矩阵乘法",
            ha="center", va="center", fontsize=9.3, color=C_TEXT)
    ax.text(13.15, 7.50, "位置数不变\n最后一维变成词表大小",
            ha="center", va="center", fontsize=8.8, color=C_TEXT)

    panel(ax, (15.15, 7.0), 8.35, 5.35, "④ logits · 每个位置都有一整排候选词分数",
          fc="#fdfbfe", ec=C_PURPLE)
    grid_x, grid_y = 17.0, 8.05
    cell_w, cell_h = 1.05, 1.05
    for c, word in enumerate(words):
        ax.text(grid_x + (c + 0.5) * cell_w, grid_y + 2 * cell_h + 0.18,
                word, ha="center", va="bottom", fontsize=8.5, color=C_TEXT)
    for r in range(2):
        display_r = 1 - r
        ax.text(grid_x - 0.18, grid_y + (display_r + 0.5) * cell_h,
                f"位置 {r}", ha="right", va="center", fontsize=9,
                color=C_PURPLE if r == 1 else C_TEXT,
                fontweight="bold" if r == 1 else "normal")
        winner = int(np.argmax(logits[r]))
        for c, value in enumerate(logits[r]):
            is_winner = c == winner
            is_last = r == 1
            rect = Rectangle(
                (grid_x + c * cell_w, grid_y + display_r * cell_h),
                cell_w,
                cell_h,
                facecolor=C_LIGHT_GREEN if is_winner else (C_LIGHT_PURPLE if is_last else "white"),
                edgecolor=C_GREEN if is_winner else (C_PURPLE if is_last else "#b9c0ca"),
                linewidth=2.0 if is_winner else 1.1,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(grid_x + (c + 0.5) * cell_w,
                    grid_y + (display_r + 0.5) * cell_h,
                    f"{value:.1f}", ha="center", va="center", fontsize=9,
                    color="#176b2c" if is_winner else C_DARK, zorder=4)
    ax.text(19.62, 7.52,
            "教学图用 E=4、vocab=5 真算 h @ W.T；课程默认模型是 E=64、vocab=128",
            ha="center", va="center", fontsize=8.5, color=C_TEXT)

    arrow(ax, (4.45, 9.65), (5.0, 9.65))
    arrow(ax, (11.15, 9.65), (11.7, 9.65))
    arrow(ax, (14.6, 9.65), (15.15, 9.65))

    panel(ax, (3.1, 1.25), 17.8, 4.55,
          "⑤ generate_naive 只消费最后位置 logits[0, -1, :]", fc="#fffdfa", ec=C_ORANGE)
    ax.text(5.15, 3.62, "最后位置 h₁", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_GREEN)
    ax.text(5.15, 2.82, "表示整个前缀 [我, 爱]", ha="center", va="center",
            fontsize=9.2, color=C_TEXT)
    arrow(ax, (6.5, 3.25), (7.45, 3.25), label="lm_head")
    ax.text(10.0, 3.62, "最后一排 logits", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=C_PURPLE)
    ax.text(10.0, 2.82, "[0.1, 0.4, 3.2, 0.5, -4.2]", ha="center", va="center",
            fontsize=9.2, family="monospace", color=C_TEXT)
    arrow(ax, (12.5, 3.25), (13.5, 3.25), label="argmax")
    token_box(ax, (14.0, 2.65), 1.7, 1.25, "吃", "新", fc=C_LIGHT_GREEN, ec=C_GREEN, lw=2.0)
    arrow(ax, (15.7, 3.25), (16.75, 3.25), label="拼回 ids")
    ax.text(18.5, 3.6, "[我, 爱, 吃]", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_BLUE)
    ax.text(18.5, 2.72, "序列长度到这里才从 2 变成 3", ha="center", va="center",
            fontsize=9.3, color=C_TEXT)

    ax.text(12, 0.45,
            "Attention：L 个位置 → L 个向量　　lm_head：每个向量 → vocab 个分数　　generate：最后一排分数 → 1 个新 token",
            ha="center", va="center", fontsize=10.6, color=C_DARK, fontweight="bold")
    save(fig, "w2_hidden_to_logits_bridge.png")


def training_vs_generation():
    """图 2：同一位置的 logits 在生成和训练中面对不同长度输入。"""
    fig, ax = canvas(w=16, h=9.2, xlim=(0, 24), ylim=(0, 13.9))
    ax.text(12, 13.45, "同一个 logits[0, 1]：生成只给前缀，训练却一次送入整句",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#222222")
    ax.text(12, 12.86,
            "因果掩码的任务，是让右边的训练调用在位置 1 上严格等价于左边的真实生成调用",
            ha="center", va="center", fontsize=10.4, color="#555555")

    panel(ax, (0.45, 3.15), 10.9, 8.85, "A · generate_naive 当前这一轮", fc="#f9fcff", ec=C_BLUE)
    ax.text(5.9, 10.95, "A = model([我, 爱])[0, 1, :]",
            ha="center", va="center", fontsize=10.2, color=C_BLUE)
    token_box(ax, (2.05, 8.65), 2.2, 1.35, "我", 0)
    token_box(ax, (5.15, 8.65), 2.2, 1.35, "爱", 1, fc=C_LIGHT_ORANGE, ec=C_ORANGE, lw=2.2)
    ax.text(5.9, 7.88, "输入中根本没有“吃”", ha="center", va="center",
            fontsize=10.1, color=C_BLUE, fontweight="bold")
    ax.text(5.9, 6.72, "位置 1 的隐藏向量 h₁\n只能汇总 [我, 爱]",
            ha="center", va="center", fontsize=10.3, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT_GREEN, ec=C_GREEN))
    arrow(ax, (5.9, 6.05), (5.9, 5.15))
    ax.text(5.9, 4.62, "lm_head(h₁) → logits → 选出“吃”",
            ha="center", va="center", fontsize=10.3, color=C_PURPLE, fontweight="bold")
    ax.text(5.9, 3.72, "这一步是在预测，不存在偷看", ha="center", va="center",
            fontsize=9.6, color=C_TEXT)

    panel(ax, (12.65, 3.15), 10.9, 8.85, "B · 训练时的一次并行 forward", fc="#fffafa", ec=C_RED)
    ax.text(18.1, 10.95, "B = model([我, 爱, 吃, 苹果])[0, 1, :]",
            ha="center", va="center", fontsize=9.5, color=C_RED)
    starts = [13.15, 15.7, 18.25, 20.8]
    words = ["我", "爱", "吃", "苹果"]
    for i, (x, word) in enumerate(zip(starts, words)):
        if i == 1:
            token_box(ax, (x, 8.65), 1.95, 1.35, word, i,
                      fc=C_LIGHT_ORANGE, ec=C_ORANGE, lw=2.2)
        elif i == 2:
            token_box(ax, (x, 8.65), 1.95, 1.35, word, i,
                      fc=C_LIGHT_RED, ec=C_RED, lw=2.2)
        else:
            token_box(ax, (x, 8.65), 1.95, 1.35, word, i)
    ax.text(18.1, 7.84, "训练目标右移一格： [爱, 吃, 苹果, —]",
            ha="center", va="center", fontsize=10.0, color=C_TEXT)
    ax.text(16.68, 6.72, "位置 1 的 h₁\n目标正是“吃”",
            ha="center", va="center", fontsize=10.0, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT_ORANGE, ec=C_ORANGE))
    arrow(ax, (19.22, 8.65), (17.25, 7.35), color=C_RED, lw=2.3,
          label="无掩码：可读取答案 V吃", label_offset=(0.4, 0.35),
          connectionstyle="arc3,rad=0.18")
    ax.text(18.1, 5.35, "若 attention[1, 2] ≈ 1，h₁ 就几乎等于 V吃\nlm_head 再把它翻译回 token“吃”",
            ha="center", va="center", fontsize=10.0, color=C_RED,
            bbox=dict(boxstyle="round,pad=0.38", fc=C_LIGHT_RED, ec=C_RED))
    ax.text(18.1, 3.75, "训练 loss 很低，但 generate 时这条捷径不存在",
            ha="center", va="center", fontsize=9.7, color=C_RED, fontweight="bold")

    ax.text(12, 2.16, "因果掩码要求 mask[1, 2] = False → attention[1, 2] = 0",
            ha="center", va="center", fontsize=11.1, color=C_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.42", fc=C_LIGHT_GREEN, ec=C_GREEN))
    ax.text(12, 1.2,
            "有掩码：B 的位置 1 也只能读取 [我, 爱]，所以 A = B；完整训练与真实生成使用同一种条件",
            ha="center", va="center", fontsize=10.4, color=C_DARK)
    ax.text(12, 0.48,
            "没有掩码：训练算的是 P(吃 | 我, 爱, 吃, 苹果)　　有掩码：训练被迫算 P(吃 | 我, 爱)",
            ha="center", va="center", fontsize=10.2, color=C_TEXT, fontweight="bold")
    save(fig, "w2_training_vs_generation.png")


def main():
    hidden_to_logits()
    training_vs_generation()


if __name__ == "__main__":
    main()
