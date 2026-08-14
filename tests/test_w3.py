"""Week 3：KV cache——结果一致 + 内存算术。"""

import torch

from tests._impl import load

kv_mod = load("cache.kv_cache")
generate_mod = load("generate")


def test_naive_cache_append_and_len():
    cache = kv_mod.NaiveKVCache(num_layers=2)
    assert cache.seq_len() == 0
    k1 = torch.ones(1, 2, 3, 4)
    v1 = torch.zeros(1, 2, 3, 4)
    k, v = cache.append_and_get(0, k1, v1)
    assert k.shape == (1, 2, 3, 4)
    assert cache.seq_len() == 3
    k2 = torch.ones(1, 2, 1, 4) * 2
    k, v = cache.append_and_get(0, k2, k2)
    assert k.shape == (1, 2, 4, 4)
    assert cache.seq_len() == 4
    # 旧的在前、新的在后
    assert torch.equal(k[0, 0, :3], torch.ones(3, 4))
    assert torch.equal(k[0, 0, 3], torch.full((4,), 2.0))
    # 另一层还是空的
    assert cache.seq_len(1) == 0
    cache.reset()
    assert cache.seq_len() == 0


def test_kv_cache_memory_bytes():
    # 2 层、4 头、每头 8 维、10 个 token、bf16（2 字节）：
    # 2(K和V) × 4 × 8 × 2 字节 = 每 token 每层 128 字节；×2 层 ×10 token = 2560
    n = kv_mod.kv_cache_memory_bytes(2, 4, 8, 10, dtype_bytes=2)
    assert n == 2560


def test_generate_with_cache_matches_naive(mini_model):
    """带 cache 的生成必须和不带 cache 的逐 token 完全一致。"""
    ids = torch.tensor([[3, 1, 4]])
    naive = generate_mod.generate_naive(mini_model, ids, max_new_tokens=8)
    cached = generate_mod.generate_with_cache(mini_model, ids, max_new_tokens=8)
    assert naive.tolist() == cached.tolist()


def test_generate_with_cache_respects_eos(mini_model):
    """用真实模型先探出第一个生成词，再把它当 eos，应当立刻停。"""
    ids = torch.tensor([[3, 1, 4]])
    full = generate_mod.generate_with_cache(mini_model, ids, max_new_tokens=6)
    first_new = full[0, ids.shape[1]].item()
    stopped = generate_mod.generate_with_cache(
        mini_model, ids, max_new_tokens=6, eos_token_id=first_new
    )
    assert stopped.shape[1] == ids.shape[1] + 1  # 只生成了一个就停
