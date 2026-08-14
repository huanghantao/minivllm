"""Week 3：KV cache——别让模型重复算已经算过的东西。

NaiveKVCache 是最直白的实现：每层两个张量（K、V），
每来一个新 token 就在序列维度上 torch.cat 拼接。
它慢在"反复拼接"，但思想是所有 KV cache 的根。
"""

import torch


class NaiveKVCache:
    """每层一对 (K, V)，形状 (batch, heads, seq_len, head_dim)，逐 token 变长。

    用法（配合 MiniTransformer.forward 的 use_cache=True）：
        cache = NaiveKVCache(num_layers=2)
        logits, cache = model(ids, past_kv=cache, use_cache=True)
    """

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.k_list = [None] * num_layers  # 每层一个 (B, H, S, D) 或 None
        self.v_list = [None] * num_layers

    def append_and_get(self, layer_idx, k, v):
        """把该层新算的 (k, v) 拼到历史后面，存起来，返回完整 (k, v)。

        k, v: (B, H, L_new, D)。返回 (B, H, L_old + L_new, D)。
        """
        old_k = self.k_list[layer_idx]
        if old_k is not None:
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([self.v_list[layer_idx], v], dim=2)
        self.k_list[layer_idx] = k
        self.v_list[layer_idx] = v
        return k, v

    def seq_len(self, layer_idx=0):
        """该层已经缓存了多少个 token。"""
        k = self.k_list[layer_idx]
        return 0 if k is None else k.shape[2]

    def reset(self):
        """清空全部缓存。"""
        self.k_list = [None] * self.num_layers
        self.v_list = [None] * self.num_layers


def kv_cache_memory_bytes(num_layers, num_heads, head_dim, seq_len, dtype_bytes=2):
    """算一算：一个请求的 KV cache 要占多少字节？

    每个 token、每一层，都要存一份 K 和一份 V，
    每份是 num_heads × head_dim 个数，每个数占 dtype_bytes 字节。
    """
    per_token_per_layer = 2 * num_heads * head_dim * dtype_bytes  # 2 = K 和 V
    return num_layers * seq_len * per_token_per_layer
