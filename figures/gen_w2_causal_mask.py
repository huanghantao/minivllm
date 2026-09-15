"""W2 因果掩码图解：泄题、造表、执行掩码与 KV cache 偏移。"""

import pathlib
import sys

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
C_TEXT = "#3e4650"
C_LIGHT_BLUE = "#eef4fb"
C_LIGHT_GREEN = "#edf8ef"
C_LIGHT_ORANGE = "#fff5e8"
C_LIGHT_PURPLE = "#f5eff9"
C_LIGHT_RED = "#fff0f0"


def panel(ax, xy, w, h, title=None, fc="#fbfcfe", ec="#d7dde5", lw=1.3):
    """画章节图里反复使用的圆角面板。"""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.10",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=1,
    )
    ax.add_patch(patch)
    if title:
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


def token_box(ax, xy, w, h, token, position, fc=C_LIGHT_BLUE, ec=C_BLUE,
              alpha=1.0, lw=1.5):
    """一个带绝对位置编号的 token 盒子。"""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.58,
        token,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=C_DARK,
        alpha=alpha,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + 0.12,
        f"位置 {position}",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#6b7280",
        alpha=alpha,
        zorder=4,
    )


def matrix(ax, xy, values, cell_w=0.72, cell_h=0.62, colors=None,
           edges=None, row_labels=None, col_labels=None, fontsize=10,
           label_color="#606873"):
    """画一个按阅读顺序从上到下排列的矩阵。"""
    x, y = xy
    rows = len(values)
    cols = len(values[0])
    for r, row in enumerate(values):
        display_r = rows - 1 - r
        if row_labels:
            ax.text(
                x - 0.16,
                y + (display_r + 0.5) * cell_h,
                row_labels[r],
                ha="right",
                va="center",
                fontsize=8.5,
                color=label_color,
                zorder=5,
            )
        for c, value in enumerate(row):
            fc = colors[r][c] if colors else "white"
            ec = edges[r][c] if edges else "#aeb6c2"
            rect = Rectangle(
                (x + c * cell_w, y + display_r * cell_h),
                cell_w,
                cell_h,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.25,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                x + (c + 0.5) * cell_w,
                y + (display_r + 0.5) * cell_h,
                str(value),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=C_DARK,
                zorder=4,
            )
    if col_labels:
        for c, label in enumerate(col_labels):
            ax.text(
                x + (c + 0.5) * cell_w,
                y + rows * cell_h + 0.12,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=label_color,
                zorder=5,
            )


def parallel_leak():
    """图 1：整段并行计算时，较早位置为何会看见答案。"""
    fig, ax = canvas(w=16, h=7.7, xlim=(0, 24), ylim=(0, 11.8))
    ax.text(
        12,
        11.35,
        "未来 token 明明还没生成，为什么仍然需要因果掩码？",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12,
        10.76,
        "训练和 prompt 的 prefill 会把整段一次送进模型；对较早位置来说，后面的 token 已经在同一张输入表里",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#555555",
    )

    panel(
        ax,
        (0.45, 2.45),
        10.9,
        7.45,
        "没有掩码：位置 1 可以读取整段",
        fc="#fffafa",
        ec=C_RED,
    )
    panel(
        ax,
        (12.65, 2.45),
        10.9,
        7.45,
        "有因果掩码：位置 1 只能读取自己和左边",
        fc="#f9fdf9",
        ec=C_GREEN,
    )

    tokens = ["我", "爱", "吃", "苹果"]
    for panel_x in (0.45, 12.65):
        start_x = panel_x + 0.72
        for idx, token in enumerate(tokens):
            if idx == 1:
                fc, ec, lw, alpha = C_LIGHT_ORANGE, C_ORANGE, 2.2, 1.0
            elif panel_x > 12 and idx > 1:
                fc, ec, lw, alpha = C_LIGHT_RED, C_RED, 1.4, 0.52
            else:
                fc, ec, lw, alpha = C_LIGHT_BLUE, C_BLUE, 1.4, 1.0
            token_box(
                ax,
                (start_x + idx * 2.45, 7.55),
                1.8,
                1.25,
                token,
                idx,
                fc=fc,
                ec=ec,
                alpha=alpha,
                lw=lw,
            )
        ax.text(
            panel_x + 5.45,
            9.17,
            "同一份输入 ids = [我, 爱, 吃, 苹果]",
            ha="center",
            va="center",
            fontsize=9.2,
            color=C_TEXT,
        )

    # 左侧：Q(爱) 能读取所有 key，其中“吃”就是要预测的答案。
    q_left = (5.90, 4.08)
    ax.text(
        q_left[0],
        q_left[1],
        "位置 1 的 Q\n当前 token = 爱\n任务：预测下一个 token",
        ha="center",
        va="center",
        fontsize=10,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.48", fc=C_LIGHT_ORANGE, ec=C_ORANGE, lw=1.8),
        zorder=5,
    )
    left_centers = [(2.07 + i * 2.45, 7.55) for i in range(4)]
    for idx, center in enumerate(left_centers):
        color = C_RED if idx > 1 else C_BLUE
        arrow(
            ax,
            (q_left[0], q_left[1] + 0.62),
            center,
            color=color,
            lw=1.55,
            connectionstyle=f"arc3,rad={(idx - 1.5) * 0.10}",
        )
    ax.text(
        6.98,
        6.65,
        "答案“吃”已经在下一格：泄题",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#a61b1b",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc=C_LIGHT_RED, ec=C_RED),
        zorder=6,
    )
    ax.text(
        5.9,
        3.02,
        "若未来内容混进位置 1 的表示，模型可以抄答案，学不到真正的“根据前缀预测”",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#8d2525",
    )

    # 右侧：同一个 Q 只连向 j <= i 的两个 key。
    q_right = (18.10, 4.08)
    ax.text(
        q_right[0],
        q_right[1],
        "位置 1 的 Q\n当前 token = 爱\n任务：预测下一个 token",
        ha="center",
        va="center",
        fontsize=10,
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.48", fc=C_LIGHT_ORANGE, ec=C_ORANGE, lw=1.8),
        zorder=5,
    )
    right_centers = [(14.27 + i * 2.45, 7.55) for i in range(4)]
    for idx in (0, 1):
        arrow(
            ax,
            (q_right[0], q_right[1] + 0.62),
            right_centers[idx],
            color=C_GREEN,
            lw=1.7,
            connectionstyle=f"arc3,rad={(idx - 0.5) * 0.12}",
        )
    for idx in (2, 3):
        cx = right_centers[idx][0]
        ax.text(
            cx,
            6.67,
            "× 禁止",
            ha="center",
            va="center",
            fontsize=9.2,
            color=C_RED,
            fontweight="bold",
        )
    ax.text(
        18.1,
        3.02,
        "虽然未来 token 仍在输入张量里，但它们的信息到不了位置 1",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#286c34",
    )

    ax.text(
        12,
        1.35,
        "整段一次进模型  ≠  每个位置有权读整段；掩码隔离的是信息流，不是把未来 token 从张量里删除。",
        ha="center",
        va="center",
        fontsize=11,
        color="#744000",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", fc="#fff9ef", ec=C_ORANGE),
    )
    ax.text(
        12,
        0.55,
        "位置 i 的硬规则：只允许连接到 key 位置 j ≤ i（包括自己）。",
        ha="center",
        va="center",
        fontsize=10.5,
        color=C_TEXT,
    )
    save(fig, "w2_causal_parallel_leak.png")


def mask_construction():
    """图 2：从 i/j 坐标和广播比较造出下三角 BoolTensor。"""
    fig, ax = canvas(w=16, h=7.8, xlim=(0, 25), ylim=(0, 12))
    ax.text(
        12.5,
        11.52,
        "make_causal_mask(4)：一列 i、一行 j，广播比较出整张通行证",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12.5,
        10.92,
        "行是“谁在看”（query），列是“看谁”（key）；本例 q_len = kv_len = 4，所以 offset = 0",
        ha="center",
        va="center",
        fontsize=10.3,
        color="#555555",
    )

    panel(ax, (0.45, 2.15), 8.4, 7.8, "先把查询下标竖起来，把键下标横过来", fc="#f8fbff", ec=C_BLUE)
    ax.text(2.15, 8.93, "i = arange(4).unsqueeze(1)", ha="center", fontsize=9.5,
            color=C_BLUE, fontweight="bold")
    matrix(
        ax,
        (1.78, 5.55),
        [[0], [1], [2], [3]],
        cell_w=0.75,
        cell_h=0.62,
        colors=[[C_LIGHT_BLUE]] * 4,
        edges=[[C_BLUE]] * 4,
        fontsize=10,
    )
    ax.text(2.15, 4.98, "shape (4, 1)\n每一行的 query 下标", ha="center", va="top",
            fontsize=8.8, color=C_TEXT)

    ax.text(5.80, 8.93, "j = arange(4).unsqueeze(0)", ha="center", fontsize=9.5,
            color=C_PURPLE, fontweight="bold")
    matrix(
        ax,
        (4.16, 7.05),
        [[0, 1, 2, 3]],
        cell_w=0.82,
        cell_h=0.62,
        colors=[[C_LIGHT_PURPLE] * 4],
        edges=[[C_PURPLE] * 4],
        fontsize=10,
    )
    ax.text(5.80, 6.48, "shape (1, 4)\n每一列的 key 下标", ha="center", va="top",
            fontsize=8.8, color=C_TEXT)
    ax.text(
        4.65,
        3.55,
        "广播会把 i 沿列方向重复，\n把 j 沿行方向重复；\n于是每个格子都得到一对 (i, j)。",
        ha="center",
        va="center",
        fontsize=9.4,
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.38", fc="white", ec="#cbd3dd"),
    )

    arrow(ax, (8.85, 6.12), (10.20, 6.12), color=C_ORANGE, lw=2.0)
    ax.text(
        9.52,
        6.72,
        "逐格判断\nj ≤ i + offset",
        ha="center",
        va="center",
        fontsize=9.3,
        color=C_ORANGE,
        fontweight="bold",
    )

    panel(ax, (10.20, 2.15), 14.35, 7.8, "比较结果：True 允许，False 禁止", fc="#fbfcfb", ec=C_GREEN)
    values = [
        ["✓", "×", "×", "×"],
        ["✓", "✓", "×", "×"],
        ["✓", "✓", "✓", "×"],
        ["✓", "✓", "✓", "✓"],
    ]
    colors = []
    edges = []
    for r in range(4):
        color_row = []
        edge_row = []
        for c in range(4):
            allowed = c <= r
            color_row.append(C_LIGHT_GREEN if allowed else C_LIGHT_RED)
            edge_row.append(C_GREEN if allowed else C_RED)
        colors.append(color_row)
        edges.append(edge_row)
    matrix(
        ax,
        (12.15, 4.45),
        values,
        cell_w=1.23,
        cell_h=0.91,
        colors=colors,
        edges=edges,
        row_labels=["q0 · 我", "q1 · 爱", "q2 · 吃", "q3 · 苹果"],
        col_labels=["k0 · 我", "k1 · 爱", "k2 · 吃", "k3 · 苹果"],
        fontsize=14,
    )
    ax.text(14.61, 3.88, "下三角（含对角线）全部放行", ha="center", fontsize=9.2,
            color=C_GREEN, fontweight="bold")
    ax.text(19.65, 7.65, "右上角：j > i\n这些列位于 query 的未来",
            ha="center", va="center", fontsize=9.5, color="#9f2525",
            bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT_RED, ec=C_RED))
    arrow(ax, (18.82, 7.24), (16.55, 7.18), color=C_RED, lw=1.5,
          connectionstyle="arc3,rad=-0.18")
    ax.text(19.7, 5.25, "对角线：j = i\n每个 token 可以看自己", ha="center", va="center",
            fontsize=9.5, color="#286c34",
            bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT_GREEN, ec=C_GREEN))
    arrow(ax, (18.78, 5.35), (15.96, 6.20), color=C_GREEN, lw=1.5,
          connectionstyle="arc3,rad=0.16")

    ax.text(
        12.5,
        1.25,
        "(4, 1) 与 (1, 4) 比较  →  广播成 (4, 4)；返回的是 BoolTensor，不是注意力权重。",
        ha="center",
        va="center",
        fontsize=10.8,
        color="#744000",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", fc="#fff9ef", ec=C_ORANGE),
    )
    ax.text(
        12.5,
        0.46,
        "一句话读格子：mask[i, j] 在回答“第 i 个 query 能不能读取第 j 个 key？”",
        ha="center",
        va="center",
        fontsize=10.2,
        color=C_TEXT,
    )
    save(fig, "w2_causal_mask_build.png")


def mask_pipeline():
    """图 3：2x2 测试中，False 如何最终变成注意力权重 0。"""
    fig, ax = canvas(w=16, h=8.2, xlim=(0, 25), ylim=(0, 12.6))
    ax.text(
        12.5,
        12.08,
        "2×2 手算：被禁止的位置怎样从分数 2 变成权重 0",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12.5,
        11.48,
        "完全对应 test_sdpa_mask_blocks_future：Q、K 全是 1，D=4，所以四个原始分数都是 4 ÷ √4 = 2",
        ha="center",
        va="center",
        fontsize=10.1,
        color="#555555",
    )

    panel(ax, (0.40, 6.05), 4.60, 4.55, "① 原始 scores", fc="#faf7fc", ec=C_PURPLE)
    matrix(
        ax,
        (1.58, 7.20),
        [[2, 2], [2, 2]],
        cell_w=0.95,
        cell_h=0.78,
        colors=[[C_LIGHT_PURPLE] * 2 for _ in range(2)],
        edges=[[C_PURPLE] * 2 for _ in range(2)],
        row_labels=["q0", "q1"],
        col_labels=["k0", "k1"],
        fontsize=12,
    )
    ax.text(2.70, 6.52, "还没有任何位置被挡", ha="center", fontsize=8.8, color=C_TEXT)

    panel(ax, (5.55, 6.05), 4.65, 4.55, "② Boolean mask", fc="#f9fdf9", ec=C_GREEN)
    matrix(
        ax,
        (6.73, 7.20),
        [["T", "F"], ["T", "T"]],
        cell_w=0.95,
        cell_h=0.78,
        colors=[[C_LIGHT_GREEN, C_LIGHT_RED], [C_LIGHT_GREEN, C_LIGHT_GREEN]],
        edges=[[C_GREEN, C_RED], [C_GREEN, C_GREEN]],
        row_labels=["q0", "q1"],
        col_labels=["k0", "k1"],
        fontsize=12,
    )
    ax.text(7.88, 6.52, "True = 允许；False = 禁止", ha="center", fontsize=8.8, color=C_TEXT)

    panel(ax, (10.78, 6.05), 5.15, 4.55, "③ masked_fill(~mask, -∞)", fc="#fffafa", ec=C_RED)
    matrix(
        ax,
        (12.08, 7.20),
        [[2, "−∞"], [2, 2]],
        cell_w=1.02,
        cell_h=0.78,
        colors=[[C_LIGHT_PURPLE, C_LIGHT_RED], [C_LIGHT_PURPLE, C_LIGHT_PURPLE]],
        edges=[[C_PURPLE, C_RED], [C_PURPLE, C_PURPLE]],
        row_labels=["q0", "q1"],
        col_labels=["k0", "k1"],
        fontsize=12,
    )
    ax.text(13.27, 6.52, "只有 ~mask 为 True 的格子被改写", ha="center", fontsize=8.5, color=C_TEXT)

    panel(ax, (16.55, 6.05), 4.75, 4.55, "④ softmax（按行）", fc="#fffaf3", ec=C_ORANGE)
    matrix(
        ax,
        (17.72, 7.20),
        [["1.0", "0.0"], ["0.5", "0.5"]],
        cell_w=1.0,
        cell_h=0.78,
        colors=[[C_LIGHT_ORANGE, C_LIGHT_RED], [C_LIGHT_ORANGE, C_LIGHT_ORANGE]],
        edges=[[C_ORANGE, C_RED], [C_ORANGE, C_ORANGE]],
        row_labels=["q0", "q1"],
        col_labels=["k0", "k1"],
        fontsize=11,
    )
    ax.text(18.92, 6.52, "e^(−∞) = 0；每行仍然和为 1", ha="center", fontsize=8.5, color=C_TEXT)

    arrow(ax, (5.00, 8.35), (5.55, 8.35), color=C_GRAY, lw=1.6, label="配合", label_offset=(0, 0.34))
    arrow(ax, (10.20, 8.35), (10.78, 8.35), color=C_RED, lw=1.8)
    arrow(ax, (15.93, 8.35), (16.55, 8.35), color=C_ORANGE, lw=1.8)

    # 下半部分：权重真正作用到 V 后的输出。
    panel(ax, (1.10, 1.62), 7.00, 3.35, "V：两个 token 真正携带的内容", fc="#f8fbff", ec=C_BLUE)
    matrix(
        ax,
        (2.18, 2.43),
        [[1, 0, 0, 0], [0, 0, 0, 2]],
        cell_w=0.75,
        cell_h=0.68,
        colors=[[C_LIGHT_BLUE] * 4 for _ in range(2)],
        edges=[[C_BLUE] * 4 for _ in range(2)],
        row_labels=["v0", "v1"],
        col_labels=["d0", "d1", "d2", "d3"],
        fontsize=10,
    )
    ax.text(5.55, 3.11, "v0 = [1, 0, 0, 0]\nv1 = [0, 0, 0, 2]", ha="left", va="center",
            fontsize=9.1, color=C_TEXT)

    ax.text(
        10.15,
        3.28,
        "weights @ V",
        ha="center",
        va="center",
        fontsize=10.5,
        color=C_ORANGE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.38", fc=C_LIGHT_ORANGE, ec=C_ORANGE),
    )
    arrow(ax, (8.10, 3.28), (9.20, 3.28), color=C_ORANGE, lw=1.8)
    arrow(ax, (11.10, 3.28), (12.15, 3.28), color=C_GREEN, lw=1.8)

    panel(ax, (12.15, 1.62), 9.25, 3.35, "out：未来内容有没有混进来？", fc="#f9fdf9", ec=C_GREEN)
    matrix(
        ax,
        (13.42, 2.43),
        [[1, 0, 0, 0], ["0.5", 0, 0, "1.0"]],
        cell_w=0.82,
        cell_h=0.68,
        colors=[[C_LIGHT_GREEN] * 4 for _ in range(2)],
        edges=[[C_GREEN] * 4 for _ in range(2)],
        row_labels=["q0", "q1"],
        col_labels=["d0", "d1", "d2", "d3"],
        fontsize=10,
    )
    ax.text(
        18.15,
        3.44,
        "q0：1.0×v0 + 0.0×v1\n→ v1 一滴都没混进来",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#286c34",
        fontweight="bold",
    )
    ax.text(
        18.15,
        2.53,
        "q1：0.5×v0 + 0.5×v1\n→ 可以读取两份内容",
        ha="center",
        va="center",
        fontsize=9.0,
        color=C_TEXT,
    )

    ax.text(
        12.5,
        0.73,
        "关键链条：False 不是权重 0；它先让原始分数变成 −∞，softmax 才把最终权重精确变成 0。",
        ha="center",
        va="center",
        fontsize=10.8,
        color="#744000",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", fc="#fff9ef", ec=C_ORANGE),
    )
    save(fig, "w2_causal_mask_pipeline.png")


def cache_offset():
    """图 4：KV cache 下，局部 query 下标如何通过 offset 对齐全局位置。"""
    fig, ax = canvas(w=16, h=7.9, xlim=(0, 25), ylim=(0, 12.1))
    ax.text(
        12.5,
        11.62,
        "q_len = 2、kv_len = 4：新 query 的局部下标要先搬到全局位置",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        12.5,
        11.02,
        "K/V 由“2 个缓存旧词 + 本次 2 个新词”拼成；mask 的列下标 j 一直是整段的全局下标",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#555555",
    )

    panel(ax, (0.50, 2.15), 15.20, 7.90, "先对齐位置，再判断 j ≤ i + offset", fc="#f9fbfd", ec=C_BLUE)

    # 顶部 key/token 条。
    key_x = 5.20
    key_w = 1.72
    gap = 0.20
    key_tokens = ["旧词 0", "旧词 1", "新词 0", "新词 1"]
    for idx, token in enumerate(key_tokens):
        is_old = idx < 2
        fc = C_LIGHT_BLUE if is_old else C_LIGHT_ORANGE
        ec = C_BLUE if is_old else C_ORANGE
        token_box(
            ax,
            (key_x + idx * (key_w + gap), 8.10),
            key_w,
            1.15,
            token,
            idx,
            fc=fc,
            ec=ec,
            lw=1.5,
        )
    ax.text(3.95, 8.67, "全部 K/V\n列 j", ha="center", va="center", fontsize=9.5,
            color=C_PURPLE, fontweight="bold")
    ax.plot([key_x, key_x + 2 * (key_w + gap) - gap], [7.70, 7.70], color=C_BLUE, lw=2)
    ax.text(key_x + key_w + gap / 2, 7.43, "cache：2 个旧位置", ha="center", fontsize=8.5, color=C_BLUE)
    ax.plot([key_x + 2 * (key_w + gap), key_x + 4 * (key_w + gap) - gap], [7.70, 7.70],
            color=C_ORANGE, lw=2)
    ax.text(key_x + 3 * key_w + 2.5 * gap, 7.43, "本次：2 个新位置", ha="center",
            fontsize=8.5, color=C_ORANGE)

    # 查询本地位置与全局位置的换算。
    ax.text(1.08, 6.55, "本次 Q\n行 i", ha="left", va="center", fontsize=9.5,
            color=C_PURPLE, fontweight="bold")
    q_info = [
        ("新词 0", "局部 i = 0", "全局位置 = 2"),
        ("新词 1", "局部 i = 1", "全局位置 = 3"),
    ]
    for r, (token, local, global_pos) in enumerate(q_info):
        y = 6.25 - r * 1.02
        ax.text(2.35, y, token, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=C_DARK, bbox=dict(boxstyle="round,pad=0.27", fc=C_LIGHT_ORANGE, ec=C_ORANGE))
        ax.text(3.85, y, local, ha="center", va="center", fontsize=8.8, color=C_TEXT)
        arrow(ax, (4.60, y), (5.18, y), color=C_ORANGE, lw=1.4)
        ax.text(6.20, y, global_pos, ha="center", va="center", fontsize=8.8, color="#8b4a00",
                fontweight="bold")
        arrow(ax, (7.13, y), (8.45, y), color=C_GREEN, lw=1.25)

    # 2x4 掩码。
    values = [["✓", "✓", "✓", "×"], ["✓", "✓", "✓", "✓"]]
    colors = [[C_LIGHT_GREEN, C_LIGHT_GREEN, C_LIGHT_GREEN, C_LIGHT_RED],
              [C_LIGHT_GREEN, C_LIGHT_GREEN, C_LIGHT_GREEN, C_LIGHT_GREEN]]
    edges = [[C_GREEN, C_GREEN, C_GREEN, C_RED], [C_GREEN] * 4]
    matrix(
        ax,
        (8.55, 4.72),
        values,
        cell_w=1.38,
        cell_h=1.02,
        colors=colors,
        edges=edges,
        row_labels=None,
        col_labels=["j=0", "j=1", "j=2", "j=3"],
        fontsize=14,
    )
    ax.text(11.31, 4.22, "mask shape = (q_len, kv_len) = (2, 4)", ha="center",
            fontsize=9.0, color=C_TEXT)

    ax.text(
        11.45,
        3.18,
        "offset = kv_len − q_len = 4 − 2 = 2",
        ha="center",
        va="center",
        fontsize=10.2,
        color=C_PURPLE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=C_LIGHT_PURPLE, ec=C_PURPLE),
    )
    ax.text(3.65, 3.02, "旧词已经发生在两个新词之前，\n所以两行都必须能读取前两列。",
            ha="center", va="center", fontsize=9.1, color=C_TEXT)

    panel(ax, (16.55, 2.15), 7.95, 7.90, "逐行代入，不需要背图形", fc="#fffaf3", ec=C_ORANGE)
    ax.text(
        20.52,
        8.70,
        "第 0 行（新词 0）",
        ha="center",
        va="center",
        fontsize=10.5,
        color=C_DARK,
        fontweight="bold",
    )
    ax.text(20.52, 8.05, "j ≤ 0 + 2  →  j ≤ 2", ha="center", fontsize=10.2, color=C_ORANGE)
    ax.text(20.52, 7.48, "放行旧 0、旧 1、自己；挡住新词 1", ha="center", fontsize=8.9, color=C_TEXT)

    ax.plot([17.55, 23.50], [6.82, 6.82], color="#e3d8c8", lw=1.1)
    ax.text(20.52, 6.27, "第 1 行（新词 1）", ha="center", fontsize=10.5,
            color=C_DARK, fontweight="bold")
    ax.text(20.52, 5.62, "j ≤ 1 + 2  →  j ≤ 3", ha="center", fontsize=10.2, color=C_ORANGE)
    ax.text(20.52, 5.05, "四个位置都在它的历史里：全部放行", ha="center", fontsize=8.9, color=C_TEXT)

    ax.plot([17.55, 23.50], [4.38, 4.38], color="#e3d8c8", lw=1.1)
    ax.text(20.52, 3.62, "最常见的 decode 特例", ha="center", fontsize=10.5,
            color=C_DARK, fontweight="bold")
    ax.text(20.52, 3.00, "q_len=1, kv_len=100 → offset=99", ha="center", fontsize=9.8,
            color=C_PURPLE)
    ax.text(20.52, 2.55, "唯一的新 query 能看 100 个已有位置", ha="center", fontsize=8.9, color=C_TEXT)

    ax.text(
        12.5,
        1.18,
        "offset 的含义不是“多放行几格”，而是把本批新 query 的局部位置换算成整段序列的绝对位置。",
        ha="center",
        va="center",
        fontsize=10.7,
        color="#744000",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42", fc="#fff9ef", ec=C_ORANGE),
    )
    ax.text(
        12.5,
        0.43,
        "本课程的调用约定：K/V 的排列始终是 [缓存历史, 本次新 token]，且 kv_len ≥ q_len。",
        ha="center",
        va="center",
        fontsize=10.0,
        color=C_TEXT,
    )
    save(fig, "w2_causal_cache_offset.png")


if __name__ == "__main__":
    parallel_leak()
    mask_construction()
    mask_pipeline()
    cache_offset()
