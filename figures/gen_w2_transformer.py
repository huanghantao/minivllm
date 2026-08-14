"""W2：MiniTransformer 结构图 + 真实注意力热力图（两张）。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from figures._common import C_BLUE, C_GRAY, C_ORANGE, arrow, box, canvas, save

import matplotlib.pyplot as plt
import torch


def transformer_map():
    fig, ax = canvas(w=7.5, h=9, xlim=(0, 10), ylim=(0, 13))

    box(ax, (2.5, 11.8), 5, 1.0, "token id: [5, 2, 9]", fc="#eef4fb", ec=C_BLUE)
    box(ax, (2.5, 10.0), 5, 1.0, "嵌入：token 嵌入 + 位置嵌入", fc="#fff3e6", ec=C_ORANGE)
    arrow(ax, (5, 11.8), (5, 11.1))

    # 堆叠的 DecoderBlock
    for i, y in enumerate([7.9, 6.0]):
        box(ax, (1.6, y), 6.8, 1.5, "", fc="#f7f7f7", ec=C_GRAY)
        ax.text(5, y + 1.15, f"第 {i} 层（共 N 层）", ha="center", fontsize=11,
                weight="bold")
        box(ax, (1.9, y + 0.15), 3.0, 0.62, "LayerNorm → 注意力 → 残差",
            fc="#eaf7ea", ec="#2ca02c", fontsize=8.5)
        box(ax, (5.1, y + 0.15), 3.0, 0.62, "LayerNorm → MLP → 残差",
            fc="#f3eaf7", ec="#9467bd", fontsize=8.5)
    ax.text(5, 5.45, "⋮", fontsize=22, ha="center")

    box(ax, (2.5, 3.6), 5, 1.0, "LayerNorm → lm_head", fc="#fff3e6", ec=C_ORANGE)
    box(ax, (2.5, 1.8), 5, 1.0, "logits：每个位置一份「下一个词」分数", fc="#eef4fb", ec=C_BLUE)
    arrow(ax, (5, 10.0), (5, 9.5))
    arrow(ax, (5, 3.6), (5, 2.9))
    ax.text(5, 0.6, "生成时只取最后一个位置的 logits", fontsize=11, ha="center",
            color="#555555")
    save(fig, "w2_transformer_map.png")


def attention_heatmap():
    """用 reference 的 MiniTransformer 跑一次真实前向，画出注意力权重。"""
    from reference.model.transformer import MiniConfig, MiniTransformer

    torch.manual_seed(0)
    model = MiniTransformer(MiniConfig(
        vocab_size=32, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=32
    ))
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    # 手动复现一次前向，把第 0 层的注意力权重截获
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))
    blk = model.blocks[0]
    h = blk.ln1(x)
    q = blk.attn._split_heads(blk.attn.q_proj(h))
    k = blk.attn._split_heads(blk.attn.k_proj(h))
    v = blk.attn._split_heads(blk.attn.v_proj(h))
    from reference.model.attention import make_causal_mask, scaled_dot_product_attention
    mask = make_causal_mask(ids.shape[1])
    _, weights = scaled_dot_product_attention(q, k, v, mask)

    words = ["我", "爱", "吃", "苹", "果", "。"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for head in range(4):
        ax = axes[head]
        w = weights[0, head].detach().numpy()
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"第 {head} 个头", fontsize=12)
        ax.set_xticks(range(6), words)
        ax.set_yticks(range(6), words)
        # 把上三角（被禁看的未来）打上灰叉
        for i in range(6):
            for j in range(6):
                if j > i:
                    ax.text(j, i, "✕", ha="center", va="center",
                            color="#999999", fontsize=9)
    fig.suptitle("因果掩码下的注意力权重：每个词只能看见自己和左边（✕ = 被禁止）",
                 fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.8, label="注意力权重")
    save(fig, "w2_attention_heatmap.png")


if __name__ == "__main__":
    transformer_map()
    attention_heatmap()
