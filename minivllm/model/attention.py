"""Week 2：注意力——Transformer 的心脏。

两个纯函数：
- make_causal_mask：造一张"不许偷看未来"的表格；
- scaled_dot_product_attention：注意力的全部数学，就这四行。
"""

import math

import torch


def make_causal_mask(q_len, kv_len=None):
    """造因果掩码（causal mask）：第 i 个查询只能看 ≤ 它位置的那些键。

    参数：
        q_len: 查询（query）的个数，也就是本次前向新处理的 token 数。
        kv_len: 键值（key/value）的个数，默认等于 q_len。
                有 KV cache 时 kv_len > q_len：之前缓存的词都能看。

    返回：
        形状 (q_len, kv_len) 的 BoolTensor，True 表示"允许看"。

    例：q_len=2, kv_len=4（缓存了 2 个旧词，本次新来 2 个）：
        [[True, True, True,  False],   # 新词 0 能看旧词 0、1 和自己
         [True, True, True,  True ]]   # 新词 1 能看旧词 0、1 和新词 0、自己
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # make_causal_mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """缩放点积注意力：softmax(QK^T / sqrt(d)) V。

    参数：
        q: (batch, heads, q_len, head_dim)
        k: (batch, heads, kv_len, head_dim)
        v: (batch, heads, kv_len, head_dim)
        mask: 可广播到 (batch, heads, q_len, kv_len) 的 BoolTensor，
              True 表示允许注意力流过；None 表示全允许。

    返回：
        (out, weights)
        out:     (batch, heads, q_len, head_dim)，注意力加权后的值
        weights: (batch, heads, q_len, kv_len)，注意力权重（每行和为 1）
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # scaled_dot_product_attention
