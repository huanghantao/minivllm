"""Week 1 / Week 3：生成循环。

- generate_naive：Week 1 的最朴素生成——每多一个词，就把整段话重新算一遍。
- generate_with_cache：Week 3 的升级——用 KV cache 避免重复计算。

两个函数都不认识"分词器"，只认 token id 序列（一个一维 LongTensor），
这样它们既能套在 HuggingFace 模型上，也能套在我们手写的模型上。
"""

import torch


@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens, eos_token_id=None):
    """最朴素的自回归生成：每一步都对完整序列做一次前向。

    参数：
        model: 满足 model(input_ids) -> logits 的模型（或带 .logits 的输出）。
               input_ids 形状 (1, seq_len)，logits 形状 (1, seq_len, vocab_size)。
        input_ids: 提示词的 token id，形状 (1, seq_len) 的 LongTensor。
        max_new_tokens: 最多新生成多少个 token。
        eos_token_id: 结束符 id；生成到它就提前停下。None 表示不提前停。

    返回：
        完整序列（提示词 + 新生成），形状 (1, seq_len + n) 的 LongTensor。
    """
    ids = input_ids
    for _ in range(max_new_tokens):
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        next_token = torch.argmax(logits[0, -1, :]).item()
        ids = torch.cat(
            [ids, torch.tensor([[next_token]], dtype=ids.dtype, device=ids.device)],
            dim=1,
        )
        if eos_token_id is not None and next_token == eos_token_id:
            break
    return ids


@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens, eos_token_id=None):
    """带 KV cache 的生成：prefill 一次，之后每步只算新 token。

    要求模型的前向签名是：
        logits, past_kv = model(input_ids, past_kv=None, use_cache=True)
    其中 past_kv 是 cache.NaiveKVCache 之类的"每层 K/V 仓库"。

    返回完整序列（提示词 + 新生成），形状 (1, seq_len + n)。
    """
    # ---- prefill：一次算完整个提示词，顺手把每层的 K/V 存进仓库 ----
    logits, past_kv = model(input_ids, past_kv=None, use_cache=True)
    next_token = torch.argmax(logits[0, -1, :]).item()
    out_ids = [next_token]

    # ---- decode：每步只把"刚生成的这一个 token"喂进去 ----
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_token == eos_token_id:
            break
        step_ids = torch.tensor(
            [[next_token]], dtype=input_ids.dtype, device=input_ids.device
        )
        logits, past_kv = model(step_ids, past_kv=past_kv, use_cache=True)
        next_token = torch.argmax(logits[0, -1, :]).item()
        out_ids.append(next_token)

    new_part = torch.tensor(
        [out_ids], dtype=input_ids.dtype, device=input_ids.device
    )
    return torch.cat([input_ids, new_part], dim=1)


def count_params(model):
    """数一数模型有多少个参数（Week 1 用来感受"模型有多大"）。"""
    return sum(p.numel() for p in model.parameters())
