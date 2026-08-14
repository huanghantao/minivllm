"""生图脚本公共设施：中文字体、输出目录、常用画图助手。

每个 gen_*.py 脚本开头：

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from figures._common import OUT, DATA, save, C_BLUE, ...

脚本可以独立运行，也可以被 gen_all.py 统一运行。
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"
DATA = ROOT / "figures" / "data"  # 实测数据（由 scripts/ 下的基准脚本生成）

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "STHeiti",
    "Songti SC",
]
plt.rcParams["axes.unicode_minus"] = False

C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_GRAY = "#7f7f7f"
C_PURPLE = "#9467bd"


def save(fig, name: str, dpi: int = 150):
    """保存图片到 figures/out/，并关闭画布。"""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def load_data(name):
    """读取 figures/data/<name>（基准实测数据 JSON）。"""
    with open(DATA / name) as f:
        return json.load(f)


def box(ax, xy, w, h, text, fc="#eef4fb", ec=C_BLUE, fontsize=11, lw=1.5):
    """画一个带文字的圆角方框（流程图用）。返回中心点。"""
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec, lw=lw, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=4)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color="#333333", lw=1.8, label=None, label_offset=(0, 0.12),
          connectionstyle="arc3,rad=0"):
    """在两个点之间画箭头（流程图用）。"""
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=connectionstyle,
            shrinkA=2, shrinkB=2, mutation_scale=16,
        ),
    )
    if label is not None:
        mx, my = (p1[0] + p2[0]) / 2 + label_offset[0], (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=10, color=color, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                zorder=5)


def canvas(w=8, h=4.5, xlim=(0, 10), ylim=(0, 6)):
    """一张空白画布（无坐标轴），画流程图/示意图用。"""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax
