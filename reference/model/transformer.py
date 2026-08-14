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
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if past_kv is not None:
            # 把本次的 K/V 存进仓库，并取出"含历史"的完整 K/V
            k, v = past_kv.append_and_get(self.layer_idx, k, v)
        mask = make_causal_mask(q.shape[2], k.shape[2]).to(x.device)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        return self.o_proj(self._merge_heads(out))

    # ---- Week 4/5：分页路径（引擎接口）----

    def prefill(self, x, kv_cache, slot_mapping, block_table):
        """处理一个请求的整段提示词。x: (L, hidden)。

        1) 算出每个 token 的 K/V，按 slot_mapping 写进分页仓库；
        2) 再从仓库里把 K/V 按块表"读回来"（gather）；
        3) 做带因果掩码的注意力。
        返回 (L, hidden)。
        """
        L = x.shape[0]
        q = self.q_proj(x).view(L, self.num_heads, self.head_dim)  # (L, H, D)
        k = self.k_proj(x).view(L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(L, self.num_heads, self.head_dim)

        kv_cache.write(self.layer_idx, slot_mapping, k, v)
        slots = block_table.physical_slots()  # 该请求全部 token 的物理槽位
        k_all, v_all = kv_cache.gather(self.layer_idx, slots)  # (L, H, D)

        q4 = q.permute(1, 0, 2).unsqueeze(0)  # (1, H, L, D)
        k4 = k_all.permute(1, 0, 2).unsqueeze(0)
        v4 = v_all.permute(1, 0, 2).unsqueeze(0)
        mask = make_causal_mask(L, L).to(x.device)
        out, _ = scaled_dot_product_attention(q4, k4, v4, mask)
        out = out.squeeze(0).permute(1, 0, 2).reshape(L, -1)  # (L, H*D)
        return self.o_proj(out)

    def decode_batch(self, x, kv_cache, slot_mapping, block_tables):
        """一批请求各生成一个 token。x: (B, hidden)。

        每个新 token 的 K/V 写进各自的块；再把每个请求的历史 K/V
        从各自的块里 gather 出来，右侧补 0 对齐成等长，用掩码做批量注意力。
        返回 (B, hidden)。
        """
        B = x.shape[0]
        q = self.q_proj(x).view(B, self.num_heads, 1, self.head_dim)  # (B, H, 1, D)
        k_new = self.k_proj(x).view(B, self.num_heads, self.head_dim)
        v_new = self.v_proj(x).view(B, self.num_heads, self.head_dim)
        # 写入新 token 的 K/V（write 要 (n, H, D)）
        kv_cache.write(self.layer_idx, slot_mapping, k_new, v_new)

        slot_lists = [bt.physical_slots() for bt in block_tables]
        k_all, v_all, pad_mask = kv_cache.gather_padded(self.layer_idx, slot_lists)
        # k_all: (B, Lmax, H, D) -> (B, H, Lmax, D)
        k_all = k_all.permute(0, 2, 1, 3)
        v_all = v_all.permute(0, 2, 1, 3)
        mask = pad_mask.view(B, 1, 1, -1)  # 每个请求只看自己真实存在的词

        out, _ = scaled_dot_product_attention(q, k_all, v_all, mask)
        out = out.squeeze(2).reshape(B, -1)  # (B, H*D)
        return self.o_proj(out)


class MLP(nn.Module):
    """前馈网络：先放大 mlp_ratio 倍，过 GELU，再缩回来。"""

    def __init__(self, config):
        super().__init__()
        E = config.hidden_size
        self.fc1 = nn.Linear(E, E * config.mlp_ratio)
        self.fc2 = nn.Linear(E * config.mlp_ratio, E)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    """一层 Transformer：注意力 + MLP，各配一个 LayerNorm 和残差。"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config, layer_idx)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        x = x + self.attn(self.ln1(x), past_kv=past_kv)
        x = x + self.mlp(self.ln2(x))
        return x

    def prefill(self, x, kv_cache, slot_mapping, block_table):
        x = x + self.attn.prefill(self.ln1(x), kv_cache, slot_mapping, block_table)
        x = x + self.mlp(self.ln2(x))
        return x

    def decode_batch(self, x, kv_cache, slot_mapping, block_tables):
        x = x + self.attn.decode_batch(self.ln1(x), kv_cache, slot_mapping, block_tables)
        x = x + self.mlp(self.ln2(x))
        return x


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
        if use_cache and past_kv is None:
            past_kv = NaiveKVCache(self.config.num_layers)
        offset = past_kv.seq_len() if past_kv is not None else 0
        B, L = input_ids.shape
        positions = torch.arange(offset, offset + L, device=input_ids.device)
        x = self._embed(input_ids, positions)
        for block in self.blocks:
            x = block(x, past_kv=past_kv)
        logits = self._head(x)
        if use_cache:
            return logits, past_kv
        return logits

    # ---- Week 4/5：引擎接口 ----

    @torch.no_grad()
    def prefill(self, input_ids, kv_cache, slot_mapping, block_table):
        """一个请求的 prefill。input_ids: (L,)。返回最后一个位置的 logits (vocab,)。"""
        L = input_ids.shape[0]
        positions = torch.arange(L, device=input_ids.device)
        x = self._embed(input_ids, positions)
        for block in self.blocks:
            x = block.prefill(x, kv_cache, slot_mapping, block_table)
        return self._head(x[-1])

    @torch.no_grad()
    def decode_batch(self, token_ids, positions, kv_cache, slot_mapping, block_tables):
        """一批请求各走一步。token_ids: (B,)，positions: (B,)。

        返回 (B, vocab) 的 logits。
        """
        x = self._embed(token_ids, positions)
        for block in self.blocks:
            x = block.decode_batch(x, kv_cache, slot_mapping, block_tables)
        return self._head(x)
