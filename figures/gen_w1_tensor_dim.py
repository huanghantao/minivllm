"""W1：用“谁留下、谁被合并”解释 sum/mean 的 dim 参数。

生成：
- w1_tensor_dim_sum.png：每行求和与每列平均的简化对照图。
"""

import pathlib
import sys

from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GREEN, C_ORANGE, arrow, box, canvas, save

SCORES = [[80, 90, 100], [60, 70, 80]]
SUBJECTS = ["语文", "数学", "英语"]
STUDENTS = ["小明", "小红"]

INPUT_FILL = "#eef4fb"
RESULT_FILL = "#eaf7ea"
PANEL_FILL = "#fbfcfe"
TEXT = "#333333"
MUTED = "#666666"


def draw_panel(ax, x, y, w, h):
    """画一个淡色圆角面板。"""
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12",
        facecolor=PANEL_FILL,
        edgecolor="#d9e2ec",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(panel)


def draw_score_grid(ax, x0, y0, cw=1.35, ch=1.0):
    """画 2×3 成绩表，返回行中心、列中心和边界。"""
    rows = len(SCORES)
    cols = len(SCORES[0])

    for i, row in enumerate(SCORES):
        y = y0 + (rows - 1 - i) * ch
        for j, value in enumerate(row):
            x = x0 + j * cw
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cw,
                    ch,
                    facecolor=INPUT_FILL,
                    edgecolor=C_BLUE,
                    linewidth=1.5,
                    zorder=2,
                )
            )
            ax.text(
                x + cw / 2,
                y + ch / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=11.5,
                color=TEXT,
                zorder=3,
            )

    row_centers = [y0 + (rows - 1 - i + 0.5) * ch for i in range(rows)]
    col_centers = [x0 + (j + 0.5) * cw for j in range(cols)]

    for student, cy in zip(STUDENTS, row_centers):
        ax.text(x0 - 0.25, cy, student, ha="right", va="center", fontsize=10.5, color=MUTED)
    for subject, cx in zip(SUBJECTS, col_centers):
        ax.text(cx, y0 + rows * ch + 0.22, subject, ha="center", va="bottom", fontsize=10.5, color=MUTED)

    return {
        "right": x0 + cols * cw,
        "bottom": y0,
        "rows": row_centers,
        "cols": col_centers,
    }


def label_pill(ax, x, y, w, text, color, fill):
    """画“留下/合并”说明标签。"""
    box(ax, (x, y), w, 0.58, text, fc=fill, ec=color, fontsize=10.2, lw=1.3)


def main():
    fig, ax = canvas(w=12, h=6.8, xlim=(0, 16), ylim=(0, 9.1))

    ax.text(8, 8.72, "dim 决定“合并谁”", ha="center", va="center", fontsize=16, color=TEXT, weight="bold")
    ax.text(
        8,
        8.28,
        "先看谁要各自留下一个结果，再找出被合并的那一维",
        ha="center",
        va="center",
        fontsize=10.8,
        color=MUTED,
    )

    draw_panel(ax, 0.35, 0.85, 7.35, 6.95)
    draw_panel(ax, 8.3, 0.85, 7.35, 6.95)

    # 左：每名学生一个总分——留下学生，合并科目。
    ax.text(4.0, 7.35, "每名学生一个总分", ha="center", fontsize=13.2, color=TEXT, weight="bold")
    ax.text(4.0, 6.92, "每行求和：scores.sum(dim=1)", ha="center", fontsize=10.7, color=C_BLUE)
    left = draw_score_grid(ax, 1.25, 4.0)

    ax.text(5.9, 6.15, "每行 3 个数\n合成 1 个", ha="center", va="center", fontsize=10.2, color=C_ORANGE)
    for cy, result in zip(left["rows"], ["270", "210"]):
        arrow(ax, (left["right"] + 0.08, cy), (6.15, cy), color=C_ORANGE, lw=1.7)
        box(ax, (6.15, cy - 0.42), 1.0, 0.84, result, fc=RESULT_FILL, ec=C_GREEN, fontsize=11.5, lw=1.5)

    label_pill(ax, 0.9, 2.35, 3.0, "留下：学生（第 0 维）", C_GREEN, RESULT_FILL)
    label_pill(ax, 4.15, 2.35, 3.0, "合并：科目（第 1 维）", C_ORANGE, "#fff3e6")
    ax.text(4.0, 1.58, "所以 dim=1　　shape：(2, 3) → (2,)", ha="center", fontsize=10.7, color=TEXT, weight="bold")

    # 右：每门科目一个平均分——留下科目，合并学生。
    ax.text(12.0, 7.35, "每门科目一个平均分", ha="center", fontsize=13.2, color=TEXT, weight="bold")
    ax.text(12.0, 6.92, "每列平均：scores.mean(dim=0)", ha="center", fontsize=10.7, color=C_BLUE)
    right = draw_score_grid(ax, 10.0, 4.35)

    ax.text(14.75, 5.35, "每列 2 个数\n合成 1 个", ha="center", va="center", fontsize=10.2, color=C_ORANGE)
    for cx, result in zip(right["cols"], ["70", "80", "90"]):
        arrow(ax, (cx, right["bottom"] - 0.04), (cx, 3.45), color=C_ORANGE, lw=1.7)
        box(ax, (cx - 0.48, 2.62), 0.96, 0.78, result, fc=RESULT_FILL, ec=C_GREEN, fontsize=11.2, lw=1.5)

    label_pill(ax, 8.85, 1.72, 3.0, "留下：科目（第 1 维）", C_GREEN, RESULT_FILL)
    label_pill(ax, 12.1, 1.72, 3.0, "合并：学生（第 0 维）", C_ORANGE, "#fff3e6")
    ax.text(12.0, 1.08, "所以 dim=0　　shape：(2, 3) → (3,)", ha="center", fontsize=10.7, color=TEXT, weight="bold")

    ax.text(
        8,
        0.3,
        "“每行 / 每列”说的是谁留下；dim 说的是谁被合并。",
        ha="center",
        va="center",
        fontsize=11.8,
        color=TEXT,
        weight="bold",
    )

    save(fig, "w1_tensor_dim_sum.png")


if __name__ == "__main__":
    main()
