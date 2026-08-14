"""W1：自回归生成流程——"一个词一个词蹦出来"是怎么回事。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GREEN, C_ORANGE, arrow, box, canvas, save


def main():
    fig, ax = canvas(w=10, h=4.2, xlim=(0, 14), ylim=(0, 6))

    # 第一行：第 1 步
    box(ax, (0.3, 3.9), 3.0, 1.0, "巴黎是法国的首都", fc="#eef4fb", ec=C_BLUE)
    box(ax, (4.3, 3.9), 2.2, 1.0, "模型\n（一次前向）", fc="#fff3e6", ec=C_ORANGE)
    box(ax, (7.5, 3.9), 1.8, 1.0, "。", fc="#eaf7ea", ec=C_GREEN)
    arrow(ax, (3.3, 4.4), (4.3, 4.4))
    arrow(ax, (6.5, 4.4), (7.5, 4.4), label="预测下一个词")

    # 第二行：第 2 步——把上一个词接回去再算
    box(ax, (0.3, 1.6), 3.0, 1.0, "巴黎是法国的首都。", fc="#eef4fb", ec=C_BLUE)
    box(ax, (4.3, 1.6), 2.2, 1.0, "模型\n（又一次前向）", fc="#fff3e6", ec=C_ORANGE)
    box(ax, (7.5, 1.6), 1.8, 1.0, "它", fc="#eaf7ea", ec=C_GREEN)
    arrow(ax, (3.3, 2.1), (4.3, 2.1))
    arrow(ax, (6.5, 2.1), (7.5, 2.1))
    # 新词接回输入的箭头
    arrow(ax, (8.4, 3.9), (1.8, 2.7), connectionstyle="arc3,rad=0.35",
          label="把新词接回输入", label_offset=(1.2, 0.55))

    ax.text(10.1, 4.4, "每多一个词，\n就要完整重算一遍！", fontsize=13,
            color="#b03a2e", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2f0", ec="#b03a2e"))
    ax.text(7.0, 0.35, "这就是「自回归」：输出会回到输入。也是 Week 3 要解决的大浪费。",
            fontsize=11, ha="center", color="#555555")

    save(fig, "w1_autoregressive.png")


if __name__ == "__main__":
    main()
