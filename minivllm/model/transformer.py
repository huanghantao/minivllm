"""Week 2：MiniTransformer——一个能跑的最小 GPT。

这个文件是整个教程的"心脏模型"。它分三次长大：

- Week 2：只会 `forward(input_ids)`，一次前向算出每个位置的下一个词预测；
- Week 3：学会 `forward(input_ids, past_kv, use_cache=True)`，用 KV cache 提速；
- Week 4/5：再学会 `prefill` / `decode_batch` 这两个"引擎接口"，
  K/V 存进分页仓库（PagedKVCache），为调度器服务。

结构是最经典的 decoder-only Transformer（GPT 家族）：

    token 嵌入 + 位置嵌入
      → N × [ LayerNorm → 多头注意力 → 残差
              → LayerNorm → MLP → 残差 ]
      → LayerNorm → lm_head → logits
"""

import torch
import torch.nn as nn

from .attention import make_causal_mask, scaled_dot_product_attention
from ..cache.kv_cache import NaiveKVCache


class MiniConfig:
    """MiniTransformer 的尺寸表。默认值很小，CPU 上也能瞬间跑完。"""

    def __init__(
        self,
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
        mlp_ratio=4,
    ):
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.mlp_ratio = mlp_ratio


class MultiHeadAttention(nn.Module):
    """多头注意力：把 hidden_size 切成 num_heads 份，每头独立做注意力。"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        E = config.hidden_size
        self.q_proj = nn.Linear(E, E, bias=False)
        self.k_proj = nn.Linear(E, E, bias=False)
        self.v_proj = nn.Linear(E, E, bias=False)
        self.o_proj = nn.Linear(E, E, bias=False)

    # ---- 形状小工具 ----

    def _split_heads(self, t):
        """(B, L, H*D) -> (B, H, L, D)：把"人头"维度拆出来。"""
        B, L, _ = t.shape
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t):
        """(B, H, L, D) -> (B, L, H*D)：拼回去。"""
        B, H, L, D = t.shape
        return t.transpose(1, 2).reshape(B, L, H * D)

    # ---- Week 2/3：朴素路径（可选 KV cache）----

    def forward(self, x, past_kv=None):
        """x: (B, L, hidden)。返回 (B, L, hidden)。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MultiHeadAttention.forward

    # ---- Week 4/5：分页路径（引擎接口）----

    def prefill(self, x, kv_cache, slot_mapping, block_table):
        """处理一个请求的整段提示词。x: (L, hidden)。

        1) 算出每个 token 的 K/V，按 slot_mapping 写进分页仓库；
        2) 再从仓库里把 K/V 按块表"读回来"（gather）；
        3) 做带因果掩码的注意力。
        返回 (L, hidden)。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MultiHeadAttention.prefill

    def decode_batch(self, x, kv_cache, slot_mapping, block_tables):
        """一批请求各生成一个 token。x: (B, hidden)。

        每个新 token 的 K/V 写进各自的块；再把每个请求的历史 K/V
        从各自的块里 gather 出来，右侧补 0 对齐成等长，用掩码做批量注意力。
        返回 (B, hidden)。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MultiHeadAttention.decode_batch


class MLP(nn.Module):
    """前馈网络：先放大 mlp_ratio 倍，过 GELU，再缩回来。"""

    def __init__(self, config):
        super().__init__()
        E = config.hidden_size
        self.fc1 = nn.Linear(E, E * config.mlp_ratio)
        self.fc2 = nn.Linear(E * config.mlp_ratio, E)

    def forward(self, x):
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MLP.forward


class DecoderBlock(nn.Module):
    """一层 Transformer：注意力 + MLP，各配一个 LayerNorm 和残差。"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config, layer_idx)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # DecoderBlock.forward

    def prefill(self, x, kv_cache, slot_mapping, block_table):
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # DecoderBlock.prefill

    def decode_batch(self, x, kv_cache, slot_mapping, block_tables):
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # DecoderBlock.decode_batch


class MiniTransformer(nn.Module):
    """完整的小 GPT：嵌入 → N 层 → 归一化 → 输出头。"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or MiniConfig()
        C = self.config
        self.tok_emb = nn.Embedding(C.vocab_size, C.hidden_size)
        self.pos_emb = nn.Embedding(C.max_seq_len, C.hidden_size)
        self.blocks = nn.ModuleList(
            [DecoderBlock(C, i) for i in range(C.num_layers)]
        )
        self.ln_f = nn.LayerNorm(C.hidden_size)
        self.lm_head = nn.Linear(C.hidden_size, C.vocab_size, bias=False)

    def _embed(self, input_ids, positions):
        return self.tok_emb(input_ids) + self.pos_emb(positions)

    def _head(self, x):
        return self.lm_head(self.ln_f(x))

    # ---- Week 2/3 ----

    def forward(self, input_ids, past_kv=None, use_cache=False):
        """朴素前向。input_ids: (B, L) 的 LongTensor。

        use_cache=False：返回 logits (B, L, vocab)；
        use_cache=True：返回 (logits, past_kv)，past_kv 是 NaiveKVCache。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MiniTransformer.forward

    # ---- Week 4/5：引擎接口 ----

    @torch.no_grad()
    def prefill(self, input_ids, kv_cache, slot_mapping, block_table):
        """一个请求的 prefill。input_ids: (L,)。返回最后一个位置的 logits (vocab,)。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MiniTransformer.prefill

    @torch.no_grad()
    def decode_batch(self, token_ids, positions, kv_cache, slot_mapping, block_tables):
        """一批请求各走一步。token_ids: (B,)，positions: (B,)。

        返回 (B, vocab) 的 logits。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # MiniTransformer.decode_batch
