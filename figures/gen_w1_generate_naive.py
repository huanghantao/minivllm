"""W1：generate_naive 图解——单轮数据流与完整生成循环。"""

import pathlib
import sys

from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import (
    C_BLUE,
    C_GRAY,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    arrow,
    box,
    canvas,
    save,
)


def panel(ax, xy, w, h, title=None, fc="#fbfcfe", ec="#d7dde5"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.10",
        facecolor=fc,
        edgecolor=ec,
        lw=1.2,
        zorder=1,
    )
    ax.add_patch(patch)
    if title:
        ax.text(x + 0.25, y + h - 0.35, title, ha="left", va="top",
                fontsize=11, fontweight="bold", color="#333333", zorder=4)


def draw_logits_table(ax, x, y):
    panel(ax, (x, y), 5.3, 3.25, "模型输出 logits · shape (1, 2, 8)", fc="#faf7fc", ec=C_PURPLE)
    cell_w, cell_h = 0.48, 0.56
    grid_x, grid_y = x + 1.05, y + 0.68
    values = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 10, 0, 0, 0, 0]]

    for col in range(8):
        ax.text(grid_x + (col + 0.5) * cell_w, grid_y + 2 * cell_h + 0.16,
                str(col), ha="center", va="bottom", fontsize=8, color="#666666")

    for row, row_values in enumerate(values):
        display_row = 1 - row
        ax.text(grid_x - 0.15, grid_y + (display_row + 0.5) * cell_h,
                "位置 0" if row == 0 else "位置 1（最后）",
                ha="right", va="center", fontsize=8.5,
                color=C_PURPLE if row == 1 else "#666666",
                fontweight="bold" if row == 1 else "normal")
        for col, value in enumerate(row_values):
            is_last = row == 1
            is_winner = is_last and col == 3
            rect = Rectangle(
                (grid_x + col * cell_w, grid_y + display_row * cell_h),
                cell_w,
                cell_h,
                facecolor="#eaf7ea" if is_winner else ("#f3eaf7" if is_last else "white"),
                edgecolor=C_GREEN if is_winner else (C_PURPLE if is_last else "#b8bec7"),
                lw=1.8 if is_winner else 1.0,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(grid_x + (col + 0.5) * cell_w,
                    grid_y + (display_row + 0.5) * cell_h,
                    str(value), ha="center", va="center", fontsize=9,
                    color="#176b2c" if is_winner else "#333333", zorder=4)

    ax.text(x + 2.65, y + 0.24, "batch=1，图中省略最外层", ha="center", va="center",
            fontsize=8.5, color="#666666")


def one_round():
    fig, ax = canvas(w=15.5, h=7.3, xlim=(0, 24), ylim=(0, 10.5))
    ax.text(12, 10.05, "generate_naive 的一轮：一条序列怎样长出一个 token",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#222222")
    ax.text(12, 9.48, "例子来自 StubHFModel 的第 2 轮：ids = [[1, 2]]，词表大小 V = 8",
            ha="center", va="center", fontsize=10.5, color="#555555")

    box(ax, (0.4, 6.2), 2.7, 1.55, "当前完整 ids\n[[1, 2]]\nshape (1, 2)",
        fc="#eef4fb", ec=C_BLUE, fontsize=10.5)
    box(ax, (3.8, 6.2), 2.3, 1.55, "model(ids)\n对两个位置\n都计算分数",
        fc="#fff3e6", ec=C_ORANGE, fontsize=10.2)
    draw_logits_table(ax, 6.85, 5.45)
    box(ax, (12.9, 6.2), 3.85, 1.55, "最后位置的分数\n[0, 0, 0, 10, 0, 0, 0, 0]\nshape (8,)",
        fc="#f3eaf7", ec=C_PURPLE, fontsize=9.6)
    box(ax, (17.55, 6.2), 2.45, 1.55, "argmax + item\n最大分数的下标\nPython int 3",
        fc="#fff3e6", ec=C_ORANGE, fontsize=9.8)
    box(ax, (20.8, 6.2), 2.75, 1.55, "next_token\n3",
        fc="#eaf7ea", ec=C_GREEN, fontsize=11)

    arrow(ax, (3.1, 6.98), (3.8, 6.98))
    arrow(ax, (6.1, 6.98), (6.85, 6.98))
    arrow(ax, (12.15, 6.98), (12.9, 6.98), label="[0, -1, :]", label_offset=(0, 0.34))
    arrow(ax, (16.75, 6.98), (17.55, 6.98))
    arrow(ax, (20.0, 6.98), (20.8, 6.98))

    box(ax, (20.25, 3.35), 3.3, 1.35, "包成二维张量\n[[3]] · shape (1, 1)",
        fc="#eaf7ea", ec=C_GREEN, fontsize=10.2)
    box(ax, (14.9, 3.35), 4.25, 1.35, "torch.cat([ids, [[3]]], dim=1)\n沿序列维接到末尾",
        fc="#eef4fb", ec=C_BLUE, fontsize=9.8)
    box(ax, (9.85, 3.35), 3.85, 1.35, "下一轮的 ids\n[[1, 2, 3]]\nshape (1, 3)",
        fc="#eef4fb", ec=C_BLUE, fontsize=10.2)

    arrow(ax, (22.18, 6.2), (21.9, 4.7))
    arrow(ax, (20.25, 4.02), (19.15, 4.02))
    arrow(ax, (14.9, 4.02), (13.7, 4.02))

    ax.plot([1.75, 1.75, 17.0], [6.2, 5.05, 5.05], color=C_BLUE, lw=1.5, zorder=2)
    arrow(ax, (17.0, 5.05), (17.0, 4.7), color=C_BLUE, lw=1.5)
    ax.text(9.2, 5.05, "旧 ids = [[1, 2]] 也进入 torch.cat",
            ha="center", va="center", fontsize=9.8, color=C_BLUE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95),
            zorder=5)

    ax.text(12, 2.25,
            "两件事不要混：  [0, -1, :] 只负责取出一排分数；argmax 才负责选出 token id。",
            ha="center", va="center", fontsize=11.2, color="#7a3e00",
            bbox=dict(boxstyle="round,pad=0.42", fc="#fff8ef", ec=C_ORANGE))
    ax.text(12, 1.15,
            "形状主线： (1, 2)  →  (1, 2, 8)  →  (8,)  →  int 3  →  (1, 1)  →  (1, 3)",
            ha="center", va="center", fontsize=11.5, color="#444444", fontweight="bold")
    ax.text(12, 0.5, "一轮只增加一个 token；新 ids 会成为下一轮的完整输入。",
            ha="center", va="center", fontsize=10.5, color="#555555")

    save(fig, "w1_generate_naive_one_round.png")


def trace_panel(ax, xy, title, sequences, final_text, final_ec):
    x, y = xy
    panel(ax, (x, y), 9.0, 3.65, title)
    widths = [1.15, 1.55, 1.9, 2.25]
    starts = [x + 0.35, x + 2.0, x + 4.05, x + 6.45]
    for i, (sequence, start, width) in enumerate(zip(sequences, starts, widths)):
        is_final = i == len(sequences) - 1
        box(ax, (start, y + 1.55), width, 0.75, sequence,
            fc="#fdf2f0" if is_final and final_ec == C_RED else "#eef4fb",
            ec=final_ec if is_final else C_BLUE, fontsize=9.3,
            lw=1.8 if is_final else 1.3)
        if i:
            arrow(ax, (starts[i - 1] + widths[i - 1], y + 1.925),
                  (start, y + 1.925), lw=1.3)
    ax.text(x + 4.5, y + 0.72, final_text, ha="center", va="center",
            fontsize=9.7, color=final_ec, fontweight="bold")


def loop_and_exits():
    fig, ax = canvas(w=15, h=8.7, xlim=(0, 21), ylim=(0, 13))
    ax.text(10.5, 12.5, "generate_naive 的循环：先把新 token 接回去，再决定是否停",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#222222")
    ax.text(5.2, 11.85, "循环控制", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#444444")
    ax.text(16.0, 11.85, "两条真实退出路径", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#444444")

    box(ax, (2.2, 10.35), 5.4, 0.9, "ids = input_ids · shape (1, L)",
        fc="#eef4fb", ec=C_BLUE, fontsize=10.5)
    box(ax, (2.2, 8.55), 5.4, 1.0, "还有生成额度吗？\nk < max_new_tokens",
        fc="#f7f7f7", ec=C_GRAY, fontsize=10.2)
    box(ax, (2.2, 6.55), 5.4, 1.2,
        "完整 ids → model → 最后位置 logits → argmax\n本轮选出一个 next_token",
        fc="#fff3e6", ec=C_ORANGE, fontsize=10.0)
    box(ax, (2.2, 4.55), 5.4, 1.2,
        "先执行 torch.cat\nids 长度加 1，新 token 已进入结果",
        fc="#eef4fb", ec=C_BLUE, fontsize=10.0)
    box(ax, (2.2, 2.55), 5.4, 1.0,
        "启用了 EOS，且新生成的 token == EOS？",
        fc="#fdf2f0", ec=C_RED, fontsize=9.8)
    box(ax, (8.25, 8.55), 2.35, 1.0, "return ids\n额度用完",
        fc="#eaf7ea", ec=C_GREEN, fontsize=9.7)
    box(ax, (8.25, 2.55), 2.35, 1.0, "return ids\n结果包含 EOS",
        fc="#eaf7ea", ec=C_GREEN, fontsize=9.5)

    arrow(ax, (4.9, 10.35), (4.9, 9.55))
    arrow(ax, (4.9, 8.55), (4.9, 7.75), label="是", label_offset=(0.35, 0))
    arrow(ax, (4.9, 6.55), (4.9, 5.75))
    arrow(ax, (4.9, 4.55), (4.9, 3.55))
    arrow(ax, (7.6, 9.05), (8.25, 9.05), label="否", label_offset=(0, 0.3))
    arrow(ax, (7.6, 3.05), (8.25, 3.05), label="是", label_offset=(0, 0.3))
    arrow(ax, (2.2, 3.05), (2.2, 9.05), color=C_GRAY,
          label="否：带着更长的 ids 进入下一轮", label_offset=(-0.45, 0),
          connectionstyle="arc3,rad=-0.32")

    ax.text(6.25, 4.18, "顺序不能反",
            ha="center", va="center", fontsize=9.5, color=C_RED, fontweight="bold")

    trace_panel(
        ax,
        (11.45, 7.55),
        "路径 A · 命中 EOS=7，提前停止",
        ["[[1]]", "[[1, 2]]", "[[1, 2, 3]]", "[[1, 2, 3, 7]]"],
        "7 先被拼入 ids，随后 break，所以返回结果保留 7",
        C_RED,
    )
    trace_panel(
        ax,
        (11.45, 2.6),
        "路径 B · max_new_tokens=3，用完额度",
        ["[[1]]", "[[1, 2]]", "[[1, 2, 3]]", "[[1, 2, 3, 4]]"],
        "没有命中 EOS；三轮各生成一个 token，然后 for 自然结束",
        C_ORANGE,
    )

    ax.text(10.5, 0.95,
            "若 max_new_tokens=0，第一次额度检查就是“否”，模型一次也不调用，直接返回原输入。",
            ha="center", va="center", fontsize=10.3, color="#555555")
    ax.text(10.5, 0.35,
            "当前教学版只处理 batch=1；每一轮送给模型的都是“提示词 + 已生成 token”的完整序列。",
            ha="center", va="center", fontsize=10.3, color="#555555")

    save(fig, "w1_generate_naive_loop.png")


def main():
    one_round()
    loop_and_exits()


if __name__ == "__main__":
    main()
