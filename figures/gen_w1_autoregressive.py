"""W1：自回归生成流程——"一个词一个词蹦出来"是怎么回事。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GREEN, C_ORANGE, arrow, box, canvas, save


def main():
    fig, ax = canvas(w=10, h=5.0, xlim=(0, 14), ylim=(0, 7))

    # 第一行：第 1 步
    box(ax, (0.3, 4.9), 3.0, 1.0, "巴黎是法国的首都", fc="#eef4fb", ec=C_BLUE)
    box(ax, (4.3, 4.9), 2.2, 1.0, "模型\n（一次前向）", fc="#fff3e6", ec=C_ORANGE)
    box(ax, (7.5, 4.9), 1.8, 1.0, "。", fc="#eaf7ea", ec=C_GREEN)
    arrow(ax, (3.3, 5.4), (4.3, 5.4))
    arrow(ax, (6.5, 5.4), (7.5, 5.4))
    ax.text(7.0, 5.85, "预测下一个词", fontsize=10, color="#333333", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    # 第二行：第 2 步——把上一个词接回去再算
    box(ax, (0.3, 2.2), 3.0, 1.0, "巴黎是法国的首都。", fc="#eef4fb", ec=C_BLUE)
    box(ax, (4.3, 2.2), 2.2, 1.0, "模型\n（又一次前向）", fc="#fff3e6", ec=C_ORANGE)
    box(ax, (7.5, 2.2), 1.8, 1.0, "它", fc="#eaf7ea", ec=C_GREEN)
    arrow(ax, (3.3, 2.7), (4.3, 2.7))
    arrow(ax, (6.5, 2.7), (7.5, 2.7))

    # 新词接回输入：从「。」盒底垂直下来，再从两行之间的空隙绕回第二行输入盒顶
    arrow(ax, (8.4, 4.9), (8.4, 4.0))                       # 先垂直落下
    arrow(ax, (8.4, 4.0), (1.8, 3.35),                      # 再走空档回到输入
          connectionstyle="arc3,rad=0.12")
    ax.text(5.1, 3.85, "把新词接回输入", fontsize=10, color="#333333", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.95))

    ax.text(10.3, 5.4, "每多一个词，\n就要完整重算一遍！", fontsize=13,
            color="#b03a2e", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2f0", ec="#b03a2e"))
    ax.text(7.0, 0.6, "这就是「自回归」：输出会回到输入。也是 Week 3 要解决的大浪费。",
            fontsize=11, ha="center", color="#555555")

    save(fig, "w1_autoregressive.png")


if __name__ == "__main__":
    main()
