"""W1：sum/mean 的 dim 参数——「消灭一维」到底消灭了什么。

生成两张图：
- w1_tensor_subscripts.png：给每个数贴两个下标（i 是第 0 维，j 是第 1 维）；
- w1_tensor_dim_sum.png：sum(dim=1) 横着压扁每行 vs sum(dim=0) 竖着压扁每列。
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GREEN, C_ORANGE, arrow, box, canvas, save

VALUES = [[1, 2, 3], [4, 5, 6]]          # 教程 2.4 节用的 2×3 例子
IN_FC, IN_EC = "#eef4fb", C_BLUE          # 输入格子：蓝
OUT_FC, OUT_EC = "#fff3e6", C_ORANGE      # 结果格子：橙


def draw_grid(ax, x0, y0, cw, ch, values, highlight=None):
    """画一张 nrows×ncols 的数表。返回 (右边缘 x, 各行中心 y, 各列中心 x)。

    highlight=(i, j) 时，该格子用绿色高亮。
    """
    nrows, ncols = len(values), len(values[0])
    highlight = highlight or (-1, -1)
    for i in range(nrows):
        for j in range(ncols):
            x = x0 + j * cw
            y = y0 + (nrows - 1 - i) * ch   # i=0 画在最上面
            fc, ec, lw = IN_FC, IN_EC, 1.5
            if (i, j) == highlight:
                fc, ec, lw = "#eaf7ea", C_GREEN, 2.5
            box(ax, (x, y), cw, ch, str(values[i][j]), fc=fc, ec=ec,
                fontsize=11.5, lw=lw)
    row_centers = [y0 + (nrows - 1 - i + 0.5) * ch for i in range(nrows)]
    col_centers = [x0 + (j + 0.5) * cw for j in range(ncols)]
    return x0 + ncols * cw, row_centers, col_centers


def note(ax, xy, text, color="#555555", fontsize=10.5, ha="center"):
    ax.text(xy[0], xy[1], text, fontsize=fontsize, color=color, ha=ha,
            va="center")


def main():
    # ---------- 图 1：给每个数贴两个下标 ----------
    fig, ax = canvas(w=9, h=5.2, xlim=(0, 12), ylim=(0, 7.2))

    ax.text(6, 6.75, "每个数由两个下标定位：t[i, j]", fontsize=14,
            ha="center", color="#333333")

    x_right, row_c, col_c = draw_grid(ax, 4.3, 2.4, 1.7, 1.5, VALUES,
                                      highlight=(1, 2))

    # 列号 j（第 1 维）标在表格上方
    ax.text(sum(col_c) / 3, 6.15, "第 1 维 = 列号 j（横着数）", fontsize=12.5,
            color=C_ORANGE, ha="center")
    for j, cx in enumerate(col_c):
        ax.text(cx, 5.6, f"j={j}", fontsize=12, color=C_ORANGE, ha="center")

    # 行号 i（第 0 维）标在表格左侧
    note(ax, (1.7, 3.95), "第 0 维 = 行号 i\n（竖着数）", color=C_BLUE,
         fontsize=12.5)
    for i, cy in enumerate(row_c):
        ax.text(3.9, cy, f"i={i}", fontsize=12, color=C_BLUE, ha="right")

    # 高亮格子：两个下标一起定位到它
    ax.text(10.6, 1.55, "t[1, 2] 就是它：\n行号 1 + 列号 2", fontsize=10.5,
            color=C_GREEN, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#eaf7ea", ec=C_GREEN))
    arrow(ax, (10.15, 2.05), (9.25, 2.75), color=C_GREEN, lw=1.6)

    note(ax, (6, 0.6), "shape = (2, 3)：i 能取 2 个值，j 能取 3 个值"
         "——shape 的第几个数，就是第几个下标", fontsize=11.5)

    save(fig, "w1_tensor_subscripts.png")

    # ---------- 图 2：两种消灭方向 ----------
    fig, ax = canvas(w=10, h=8.2, xlim=(0, 14), ylim=(0, 12))

    # --- 上半：sum(dim=1)，横着压扁每行 ---
    ax.text(0.3, 11.35, "t.sum(dim=1)：消灭第 1 维（列号 j）", fontsize=13,
            ha="left", color="#333333")

    right_a, rows_a, _ = draw_grid(ax, 1.0, 8.15, 1.3, 1.15, VALUES)
    sums = [["1+2+3 = 6", "4+5+6 = 15"], [6.0, 15.0]]
    for i, cy in enumerate(rows_a):
        # 从行的右边缘扇出三条线，汇聚到右边的结果格子
        for off in (-0.38, 0.0, 0.38):
            arrow(ax, (right_a, cy + off), (8.3, cy), color=C_BLUE, lw=1.5)
        box(ax, (8.3, cy - 0.575), 2.0, 1.15, sums[0][i],
            fc=OUT_FC, ec=OUT_EC, fontsize=10.5)
    note(ax, (6.7, 10.35), "每行的 3 个格子，\n横着压成 1 个", color="#555555")
    note(ax, (9.3, 7.75), "结果 tensor([6., 15.])，形状 (2,)", fontsize=10)
    note(ax, (12.1, 9.35), "被划掉的下标：j\n幸存的下标：i（行号）\n"
         "→ 按行排列，长度 = 行数\n形状 (2, 3) → (2,)", fontsize=11,
         color="#333333")

    ax.plot([0.3, 13.7], [6.6, 6.6], color="#dddddd", lw=1, ls="--")

    # --- 下半：sum(dim=0)，竖着压扁每列 ---
    ax.text(0.3, 6.05, "t.sum(dim=0)：消灭第 0 维（行号 i）", fontsize=13,
            ha="left", color="#333333")

    _, rows_b, cols_b = draw_grid(ax, 1.0, 3.2, 1.3, 1.15, VALUES)
    means = ["1+4 = 5", "2+5 = 7", "3+6 = 9"]
    bottom_b = 3.2
    for j, cx in enumerate(cols_b):
        # 从列的底边缘扇出两条线，汇聚到下边的结果格子
        for off in (-0.38, 0.38):
            arrow(ax, (cx + off, bottom_b), (cx, 2.75), color=C_BLUE, lw=1.5)
        box(ax, (cx - 0.85, 1.6), 1.7, 1.15, means[j],
            fc=OUT_FC, ec=OUT_EC, fontsize=10)
    note(ax, (7.0, 4.45), "每列的 2 个格子，\n竖着压成 1 个", color="#555555")
    note(ax, (2.95, 1.05), "结果 tensor([5., 7., 9.])，形状 (3,)", fontsize=10)
    note(ax, (10.2, 2.35), "被划掉的下标：i\n幸存的下标：j（列号）\n"
         "→ 按列排列，长度 = 列数\n形状 (2, 3) → (3,)", fontsize=11,
         color="#333333")
    ax.annotate("", xy=(8.35, 2.35), xytext=(5.6, 2.35),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.5,
                                mutation_scale=14))

    save(fig, "w1_tensor_dim_sum.png")


if __name__ == "__main__":
    main()
