"""W2：注意力代码图解——手算数据流与通用张量形状。"""

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
        ax.text(
            x + 0.22,
            y + h - 0.28,
            title,
            ha="left",
            va="top",
            fontsize=10.8,
            fontweight="bold",
            color="#333333",
            zorder=4,
        )


def matrix(ax, xy, values, cell_w=0.55, cell_h=0.48, fc="white", ec="#aeb6c2",
           text_color="#333333", fontsize=9.5, row_labels=None, col_labels=None):
    x, y = xy
    rows = len(values)
    cols = len(values[0])
    for row, row_values in enumerate(values):
        display_row = rows - 1 - row
        if row_labels:
            ax.text(
                x - 0.12,
                y + (display_row + 0.5) * cell_h,
                row_labels[row],
                ha="right",
                va="center",
                fontsize=8,
                color="#666666",
            )
        for col, value in enumerate(row_values):
            rect = Rectangle(
                (x + col * cell_w, y + display_row * cell_h),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor=ec,
                lw=1.1,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                x + (col + 0.5) * cell_w,
                y + (display_row + 0.5) * cell_h,
                str(value),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                zorder=4,
            )
    if col_labels:
        for col, label in enumerate(col_labels):
            ax.text(
                x + (col + 0.5) * cell_w,
                y + rows * cell_h + 0.08,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#666666",
            )


def one_query_example():
    fig, ax = canvas(w=15.8, h=7.8, xlim=(0, 24), ylim=(0, 11.5))
    ax.text(
        12,
        11.05,
        "注意力的一次完整计算：1 个查询怎样从 2 份内容里取信息",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12,
        10.48,
        "完全对应 test_sdpa_hand_computed：Lq=1，Lkv=2，D=2，mask=None",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )

    panel(ax, (0.35, 5.65), 6.5, 4.1, "输入 Q、K，并把 K 的最后两维转置", fc="#f7fbff", ec=C_BLUE)
    ax.text(0.92, 8.65, "Q", fontsize=10, fontweight="bold", color=C_BLUE, ha="center")
    matrix(ax, (0.37, 7.55), [[1, 0]], fc="#eef4fb", ec=C_BLUE, col_labels=["d0", "d1"])
    ax.text(0.92, 6.72, "shape (1, 2)", fontsize=8.3, color="#555555", ha="center")

    ax.text(2.75, 8.65, "K", fontsize=10, fontweight="bold", color=C_BLUE, ha="center")
    matrix(
        ax,
        (2.2, 7.05),
        [[1, 0], [0, 1]],
        fc="#eef4fb",
        ec=C_BLUE,
        row_labels=["k0", "k1"],
        col_labels=["d0", "d1"],
    )
    ax.text(2.75, 6.48, "shape (2, 2)", fontsize=8.3, color="#555555", ha="center")

    arrow(ax, (3.55, 7.55), (4.2, 7.55), color=C_ORANGE, lw=1.5)
    ax.text(3.87, 8.05, "transpose\n(-2, -1)", fontsize=8.1, color=C_ORANGE, ha="center")

    ax.text(5.4, 8.65, "K^T", fontsize=10, fontweight="bold", color=C_PURPLE, ha="center")
    matrix(
        ax,
        (4.85, 7.05),
        [[1, 0], [0, 1]],
        fc="#f3eaf7",
        ec=C_PURPLE,
        row_labels=["d0", "d1"],
        col_labels=["k0", "k1"],
    )
    ax.text(5.4, 6.48, "shape (2, 2)", fontsize=8.3, color="#555555", ha="center")
    ax.text(
        3.6,
        5.82,
        "这组 K 恰好是单位阵：数值看似没变，但行列角色已经交换",
        fontsize=7.8,
        color="#666666",
        ha="center",
        va="bottom",
    )

    panel(ax, (7.7, 6.3), 4.1, 3.45, "① 打分并缩放", fc="#faf7fc", ec=C_PURPLE)
    ax.text(9.75, 8.72, "Q @ K^T ÷ √D", ha="center", fontsize=11, color=C_PURPLE, fontweight="bold")
    ax.text(9.75, 8.05, "[1, 0] @ [[1, 0], [0, 1]]", ha="center", fontsize=9, color="#444444")
    ax.text(9.75, 7.53, "= [1, 0] ÷ √2", ha="center", fontsize=9.2, color="#444444")
    matrix(ax, (9.2, 6.78), [["0.707", "0"]], fc="#f3eaf7", ec=C_PURPLE, fontsize=9)
    ax.text(9.75, 6.48, "scores · shape (1, 2)", ha="center", fontsize=8.3, color="#555555")

    panel(ax, (12.65, 6.3), 4.6, 3.45, "③ 沿 key 维做 softmax", fc="#fffaf2", ec=C_ORANGE)
    ax.text(14.95, 8.68, "softmax(scores, dim=-1)", ha="center", fontsize=10.5, color=C_ORANGE, fontweight="bold")
    ax.text(14.95, 8.0, "[0.707, 0]", ha="center", fontsize=9.6, color="#444444")
    arrow(ax, (14.95, 7.77), (14.95, 7.4), color=C_ORANGE, lw=1.3)
    matrix(ax, (14.25, 6.78), [["0.670", "0.330"]], cell_w=0.7, fc="#fff3e6", ec=C_ORANGE, fontsize=9)
    ax.text(14.95, 6.48, "weights · 每一行之和 = 1", ha="center", fontsize=8.3, color="#555555")

    arrow(ax, (6.85, 8.02), (7.7, 8.02), color=C_PURPLE, label="Q @ K^T ÷ √2", label_offset=(0, 0.34))
    arrow(ax, (11.8, 8.02), (12.65, 8.02), color=C_ORANGE)

    panel(ax, (4.0, 2.35), 4.4, 2.65, "V：真正要取走的内容", fc="#f7fbff", ec=C_BLUE)
    matrix(
        ax,
        (5.48, 3.05),
        [[10, 0], [0, 20]],
        cell_w=0.68,
        cell_h=0.5,
        fc="#eef4fb",
        ec=C_BLUE,
        row_labels=["v0", "v1"],
        col_labels=["d0", "d1"],
    )
    ax.text(6.16, 2.62, "shape (2, 2)", ha="center", fontsize=8.3, color="#555555")

    box(
        ax,
        (10.0, 2.92),
        3.5,
        1.45,
        "④ weights @ V\n0.670×v0 + 0.330×v1",
        fc="#fff3e6",
        ec=C_ORANGE,
        fontsize=9.6,
    )
    panel(ax, (15.05, 2.35), 4.7, 2.65, "输出：这个查询的新表示", fc="#f5fbf5", ec=C_GREEN)
    matrix(ax, (16.57, 3.25), [["6.70", "6.60"]], cell_w=0.78, fc="#eaf7ea", ec=C_GREEN, fontsize=9.2)
    ax.text(17.35, 2.78, "out · shape (1, 2)", ha="center", fontsize=8.5, color="#555555")

    arrow(ax, (8.4, 3.65), (10.0, 3.65), color=C_BLUE, label="V", label_offset=(0, 0.28))
    arrow(ax, (13.5, 3.65), (15.05, 3.65), color=C_GREEN)
    arrow(
        ax,
        (15.0, 6.3),
        (11.75, 4.37),
        color=C_ORANGE,
        label="weights",
        label_offset=(0.25, 0.15),
        connectionstyle="arc3,rad=0.16",
    )

    ax.text(
        12,
        1.35,
        "transpose 只改变 K 的行列角色；softmax 才把分数变成比例；第二次矩阵乘才按比例取 V。",
        ha="center",
        va="center",
        fontsize=11,
        color="#7a3e00",
        bbox=dict(boxstyle="round,pad=0.42", fc="#fff8ef", ec=C_ORANGE),
    )
    ax.text(
        12,
        0.55,
        "形状主线：Q (1,2) @ K^T (2,2) → scores/weights (1,2)；weights (1,2) @ V (2,2) → out (1,2)",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#444444",
        fontweight="bold",
    )

    save(fig, "w2_attention_one_query.png")


def shape_flow():
    fig, ax = canvas(w=16, h=7.6, xlim=(0, 25), ylim=(0, 11.5))
    ax.text(
        12.5,
        11.02,
        "从一张二维表推广到真实四维张量：注意力的形状主线",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12.5,
        10.42,
        "B×H 代表 batch 中每句话的每个头：这些矩阵彼此独立、并行地做同一套计算",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555555",
    )

    panel(ax, (0.35, 5.35), 5.3, 4.35, "准备最后两维", fc="#f7fbff", ec=C_BLUE)
    ax.text(0.75, 8.75, "Q", fontsize=10.5, color=C_BLUE, fontweight="bold")
    ax.text(2.45, 8.75, "(B, H, Lq, D)", fontsize=10.5, color="#333333", fontweight="bold")
    ax.text(0.75, 7.9, "K", fontsize=10.5, color=C_BLUE, fontweight="bold")
    ax.text(2.45, 7.9, "(B, H, Lkv, D)", fontsize=10.5, color="#333333", fontweight="bold")
    arrow(ax, (2.9, 7.55), (2.9, 6.95), color=C_ORANGE, lw=1.4)
    ax.text(3.25, 7.25, "transpose(-2, -1)", fontsize=8.6, color=C_ORANGE, ha="left", va="center")
    ax.text(0.75, 6.55, "K^T", fontsize=10.5, color=C_PURPLE, fontweight="bold")
    ax.text(2.45, 6.55, "(B, H, D, Lkv)", fontsize=10.5, color=C_PURPLE, fontweight="bold")
    ax.text(2.95, 5.72, "只交换最后两维；B、H 不动", fontsize=9, color="#555555", ha="center")

    panel(ax, (6.45, 5.35), 4.35, 4.35, "① Q @ K^T：收缩 D", fc="#faf7fc", ec=C_PURPLE)
    ax.text(8.62, 8.5, "(Lq, D) @ (D, Lkv)", ha="center", fontsize=10.8, color="#333333", fontweight="bold")
    ax.text(8.62, 7.72, "D 对齐并消失", ha="center", fontsize=10, color=C_RED, fontweight="bold")
    arrow(ax, (8.62, 7.45), (8.62, 6.95), color=C_PURPLE, lw=1.5)
    ax.text(8.62, 6.55, "scores", ha="center", fontsize=10.5, color=C_PURPLE, fontweight="bold")
    ax.text(8.62, 6.05, "(B, H, Lq, Lkv)", ha="center", fontsize=10.5, color=C_PURPLE, fontweight="bold")
    arrow(ax, (5.65, 7.52), (6.45, 7.52), color=C_PURPLE)

    panel(ax, (11.6, 5.35), 6.0, 4.35, "② 缩放 / mask / ③ softmax：形状不变", fc="#fffaf2", ec=C_ORANGE)
    ax.text(14.6, 8.62, "scores ÷ √D", ha="center", fontsize=10.5, color="#444444", fontweight="bold")
    ax.text(14.6, 7.92, "mask 广播到 (B, H, Lq, Lkv)", ha="center", fontsize=9.6, color=C_RED)
    ax.text(14.6, 7.22, "softmax(dim=-1)", ha="center", fontsize=10.5, color=C_ORANGE, fontweight="bold")
    ax.text(14.6, 6.6, "每个 query 沿 Lkv 个 key 横向归一", ha="center", fontsize=9.4, color="#555555")
    ax.text(14.6, 5.95, "weights  (B, H, Lq, Lkv)", ha="center", fontsize=10.5, color=C_ORANGE, fontweight="bold")
    arrow(ax, (10.8, 7.52), (11.6, 7.52), color=C_ORANGE)

    panel(ax, (18.4, 5.35), 6.2, 4.35, "④ weights @ V：收缩 Lkv", fc="#f5fbf5", ec=C_GREEN)
    ax.text(21.5, 8.6, "V  (B, H, Lkv, D)", ha="center", fontsize=10.3, color=C_BLUE, fontweight="bold")
    ax.text(21.5, 7.84, "(Lq, Lkv) @ (Lkv, D)", ha="center", fontsize=10.8, color="#333333", fontweight="bold")
    ax.text(21.5, 7.13, "Lkv 对齐并消失", ha="center", fontsize=10, color=C_RED, fontweight="bold")
    arrow(ax, (21.5, 6.92), (21.5, 6.5), color=C_GREEN, lw=1.5)
    ax.text(21.5, 6.08, "out  (B, H, Lq, D)", ha="center", fontsize=10.8, color=C_GREEN, fontweight="bold")
    arrow(ax, (17.6, 7.52), (18.4, 7.52), color=C_GREEN)

    panel(ax, (1.25, 1.85), 10.7, 2.25, "两次矩阵乘，分别“消灭”一个维度", fc="#fbfcfe", ec=C_GRAY)
    ax.text(6.6, 3.25, "第一次： (Lq, D) @ (D, Lkv)  →  (Lq, Lkv)", ha="center", fontsize=10.4, color=C_PURPLE, fontweight="bold")
    ax.text(6.6, 2.55, "第二次： (Lq, Lkv) @ (Lkv, D)  →  (Lq, D)", ha="center", fontsize=10.4, color=C_GREEN, fontweight="bold")

    panel(ax, (13.05, 1.85), 10.7, 2.25, "普通自注意力只是一个特例", fc="#fbfcfe", ec=C_GRAY)
    ax.text(18.4, 3.25, "完整序列一次算完：Lq = Lkv = L", ha="center", fontsize=10.4, color="#333333", fontweight="bold")
    ax.text(18.4, 2.55, "有 KV cache 时常见：Lq = 1，Lkv = 已缓存长度 + 1", ha="center", fontsize=10.1, color="#333333")

    ax.text(
        12.5,
        0.75,
        "输出有 Lq 个位置（跟 Q 走），每个位置是 D 维内容（跟 V 的向量宽度走）。",
        ha="center",
        va="center",
        fontsize=11.2,
        color="#176b2c",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", fc="#f5fbf5", ec=C_GREEN),
    )

    save(fig, "w2_attention_shape_flow.png")


def main():
    one_query_example()
    shape_flow()


if __name__ == "__main__":
    main()
