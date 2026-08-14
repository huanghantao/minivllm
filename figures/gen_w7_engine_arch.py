"""W7：引擎架构图——minivllm 各零件怎么拼在一起。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_GREEN, C_ORANGE, C_PURPLE, C_RED, arrow, box, canvas, save


def arch():
    fig, ax = canvas(w=11, h=6.4, xlim=(0, 15), ylim=(0, 9))

    # 用户与门面
    box(ax, (0.4, 7.6), 3.0, 0.9, "用户代码", fc="#f7f7f7", ec=C_GRAY)
    box(ax, (0.4, 5.9), 3.0, 0.9, "LLM\n（加载/门面）", fc="#eef4fb", ec=C_BLUE)
    arrow(ax, (1.9, 7.6), (1.9, 6.9), label="model 名")

    # 引擎主体
    box(ax, (5.0, 5.9), 4.2, 0.9, "LLMEngine.add_request / step", fc="#fff3e6", ec=C_ORANGE)
    arrow(ax, (3.4, 6.35), (5.0, 6.35))

    # 调度器 + 块池
    box(ax, (5.0, 3.9), 2.0, 1.2, "Scheduler\nwaiting/running", fc="#eaf7ea", ec=C_GREEN)
    box(ax, (8.2, 3.9), 2.0, 1.2, "BlockPool\n块池", fc="#eaf7ea", ec=C_GREEN)
    arrow(ax, (6.0, 5.9), (6.0, 5.2), label="点名")
    arrow(ax, (7.0, 4.5), (8.2, 4.5), label="领/还块", label_offset=(0.6, 0.28))

    # 模型 + 分页缓存
    box(ax, (5.0, 1.6), 2.0, 1.2, "模型\nprefill/decode", fc="#f3eaf7", ec=C_PURPLE)
    box(ax, (8.2, 1.6), 2.0, 1.2, "PagedKVCache\n+ BlockTable", fc="#f3eaf7", ec=C_PURPLE)
    arrow(ax, (6.0, 3.9), (6.0, 2.9), label="跑一步")
    arrow(ax, (7.0, 2.2), (8.2, 2.2), label="写/读", label_offset=(0.6, 0.28))

    # 采样器与分词器
    box(ax, (11.6, 3.9), 1.9, 1.2, "Sampler\n采样器", fc="#fdf2f0", ec=C_RED)
    box(ax, (11.6, 1.6), 1.9, 1.2, "Tokenizer\ndetokenize", fc="#f7f7f7", ec=C_GRAY)
    arrow(ax, (10.2, 4.5), (11.6, 4.5), label="logits")
    arrow(ax, (10.2, 2.2), (11.6, 2.2), label="token id")

    # 回到引擎输出
    arrow(ax, (12.55, 3.9), (12.55, 6.35), color=C_GRAY, connectionstyle="arc3,rad=0")
    arrow(ax, (12.55, 6.35), (9.2, 6.35), color=C_GRAY, label="文本增量", label_offset=(0, 0.2))

    ax.text(7.5, 0.5, "引擎的心跳 = step()：点名 → prefill 新人 → 批量 decode → 采样 → 登记/下车",
            fontsize=12, ha="center", color="#555555")
    save(fig, "w7_engine_arch.png")


if __name__ == "__main__":
    arch()
