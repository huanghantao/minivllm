"""Week 4：PagedKVCache——vLLM 的灵魂组件 PagedAttention 的存储层。

NaiveKVCache 的问题：每个请求的 K/V 要一大块连续内存，
长了短了都不好安排，内存被切得七零八碎（碎片）。

PagedKVCache 的解法（和操作系统的分页一模一样）：
- 所有层的 K/V 都存在固定大小的"块"里，块大小 = block_size 个 token；
- 请求拿到哪些块、按什么顺序，全记在 BlockTable（页表）里；
- 注意力计算时，按块表把 K/V "读回来"（gather），算完结果一模一样。

这个文件只管"存和取"，不管注意力数学——数学在 model/attention.py，
两者通过 gather 出来的连续 K/V 衔接。
"""

import torch


class PagedKVCache:
    """所有请求共享的分页 K/V 仓库。

    每层一对张量：
        k_cache[layer]: (num_blocks, block_size, num_heads, head_dim)
        v_cache[layer]: 同上
    物理槽位 = 块号 × block_size + 块内偏移，
    所以"第 s 个槽位"就是 k_cache[layer].view(-1, H, D)[s]。
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        head_dim,
        num_blocks,
        block_size,
        device="cpu",
        dtype=torch.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        shape = (num_blocks, block_size, num_heads, head_dim)
        self.k_cache = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def write(self, layer_idx, slot_mapping, k, v):
        """把 n 个 token 的 K/V 写进指定槽位。

        slot_mapping: (n,) 的 LongTensor，物理槽位号；
        k, v: (n, num_heads, head_dim)。
        """
        slots = torch.as_tensor(slot_mapping, dtype=torch.long, device=k.device)
        flat_k = self.k_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
        flat_v = self.v_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
        flat_k[slots] = k.to(self.dtype)
        flat_v[slots] = v.to(self.dtype)

    def gather(self, layer_idx, slots):
        """按槽位号把 K/V 读回来，拼成连续张量。

        slots: list[int]，长度 L；
        返回 (k, v)，形状都是 (L, num_heads, head_dim)。
        """
        idx = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        flat_k = self.k_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
        flat_v = self.v_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
        return flat_k[idx], flat_v[idx]

    def gather_padded(self, layer_idx, slot_lists):
        """批量读取：一组请求各自的历史 K/V，补齐到等长。

        slot_lists: list[list[int]]，B 个请求各自的物理槽位序列。
        返回 (k, v, mask)：
            k, v: (B, max_len, num_heads, head_dim)，短请求左边对齐、右侧补 0；
            mask: (B, max_len) 的 BoolTensor，True 表示该位置是真实数据。
        """
        B = len(slot_lists)
        lens = [len(s) for s in slot_lists]
        max_len = max(lens)
        k = torch.zeros(
            B, max_len, self.num_heads, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        v = torch.zeros_like(k)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)
        for b, slots in enumerate(slot_lists):
            kk, vv = self.gather(layer_idx, slots)
            k[b, : len(slots)] = kk
            v[b, : len(slots)] = vv
            mask[b, : len(slots)] = True
        return k, v, mask
