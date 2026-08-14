"""Week 4：分页——块池、块表、分页仓库，以及"分页 == 不分页"的铁证。"""

import pytest
import torch

from tests._impl import load

pool_mod = load("cache.block_pool")
table_mod = load("cache.block_table")
paged_mod = load("cache.paged_cache")
generate_mod = load("generate")


# ---------- BlockPool ----------

def test_pool_allocate_and_free():
    pool = pool_mod.BlockPool(num_blocks=3, block_size=2)
    assert pool.num_free_blocks() == 3
    b0 = pool.allocate()
    b1 = pool.allocate()
    assert {b0, b1} == {0, 1}
    assert pool.num_free_blocks() == 1
    pool.free(b0)
    assert pool.num_free_blocks() == 2


def test_pool_exhaustion():
    pool = pool_mod.BlockPool(num_blocks=1, block_size=2)
    pool.allocate()
    with pytest.raises(pool_mod.OutOfBlocksError):
        pool.allocate()


def test_pool_double_free_raises():
    pool = pool_mod.BlockPool(num_blocks=2, block_size=2)
    b = pool.allocate()
    pool.free(b)
    with pytest.raises(ValueError):
        pool.free(b)


def test_pool_can_fit():
    pool = pool_mod.BlockPool(num_blocks=3, block_size=4)
    assert pool.can_fit(12)      # 恰好 3 块
    assert not pool.can_fit(13)  # 需要 4 块
    pool.allocate()
    assert pool.can_fit(8)
    assert not pool.can_fit(9)


# ---------- BlockTable ----------

def test_block_table_slots():
    pool = pool_mod.BlockPool(num_blocks=8, block_size=2)
    bt = table_mod.BlockTable(pool)
    # 物理槽位 = 块号 × 块大小 + 偏移；块按 0,1,2... 顺序领
    assert bt.append_slot() == 0  # 块 0 的第 0 格
    assert bt.append_slot() == 1  # 块 0 的第 1 格
    assert bt.append_slot() == 2  # 块 0 满了，领块 1
    assert bt.physical_slots() == [0, 1, 2]
    assert bt.num_tokens == 3


def test_block_table_needed_blocks():
    pool = pool_mod.BlockPool(num_blocks=8, block_size=4)
    bt = table_mod.BlockTable(pool)
    assert bt.needed_blocks(3) == 1
    assert bt.needed_blocks(4) == 1
    assert bt.needed_blocks(5) == 2
    for _ in range(4):
        bt.append_slot()
    assert bt.needed_blocks(1) == 1  # 当前块已满，再来 1 个也要新块
    assert bt.needed_blocks(0) == 0


def test_block_table_free_returns_blocks():
    pool = pool_mod.BlockPool(num_blocks=2, block_size=2)
    bt = table_mod.BlockTable(pool)
    for _ in range(3):
        bt.append_slot()
    assert pool.num_free_blocks() == 0
    bt.free()
    assert pool.num_free_blocks() == 2
    assert bt.num_tokens == 0
    assert bt.block_ids == []


# ---------- PagedKVCache ----------

def make_cache(num_layers=2, num_heads=4, head_dim=8, num_blocks=4, block_size=3):
    return paged_mod.PagedKVCache(
        num_layers, num_heads, head_dim, num_blocks, block_size
    )


def test_paged_cache_write_gather_roundtrip():
    cache = make_cache()
    k = torch.randn(3, 4, 8)
    v = torch.randn(3, 4, 8)
    slots = [5, 2, 9]  # 故意乱序：物理位置爱在哪在哪
    cache.write(0, slots, k, v)
    k2, v2 = cache.gather(0, slots)
    assert torch.equal(k2, k)
    assert torch.equal(v2, v)
    # 按另一个顺序 gather，就该按另一个顺序回来
    k3, _ = cache.gather(0, [9, 5])
    assert torch.equal(k3[0], k[2])
    assert torch.equal(k3[1], k[0])


def test_paged_cache_layers_are_independent():
    cache = make_cache()
    k = torch.randn(2, 4, 8)
    cache.write(0, [0, 1], k, k)
    k2, _ = cache.gather(1, [0, 1])
    assert torch.equal(k2, torch.zeros_like(k2))  # 第 1 层没写过


def test_gather_padded():
    cache = make_cache()
    cache.write(0, [0, 1, 2, 3], torch.ones(4, 4, 8), torch.ones(4, 4, 8))
    k, v, mask = cache.gather_padded(0, [[0, 1, 2], [3]])
    assert k.shape == (2, 3, 4, 8)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert torch.equal(k[1, 0], torch.ones(4, 8))
    assert torch.equal(k[1, 1], torch.zeros(4, 8))


# ---------- 铁证：分页生成 == 朴素生成 ----------

def paged_generate_single(model, prompt_ids, max_new_tokens, pool, kv_cache):
    """用分页接口做贪心生成（单请求），返回生成的 token id 列表。"""
    bt = table_mod.BlockTable(pool)
    slots = [bt.append_slot() for _ in prompt_ids]
    ids = torch.tensor(prompt_ids, dtype=torch.long)
    logits = model.prefill(ids, kv_cache, slots, bt)
    out = [int(torch.argmax(logits).item())]
    for _ in range(max_new_tokens - 1):
        slot = bt.append_slot()
        token = torch.tensor([out[-1]], dtype=torch.long)
        pos = torch.tensor([bt.num_tokens - 1], dtype=torch.long)
        logits = model.decode_batch(token, pos, kv_cache, [slot], [bt])[0]
        out.append(int(torch.argmax(logits).item()))
    return out


def test_paged_equals_naive(mini_model):
    """换存储方式绝不能换结果：分页生成的词和 Week 3 的一模一样。"""
    prompt = [3, 1, 4]
    n = 6
    cached = generate_mod.generate_with_cache(
        mini_model, torch.tensor([prompt]), max_new_tokens=n
    )
    expected = cached[0, len(prompt):].tolist()

    pool = pool_mod.BlockPool(num_blocks=8, block_size=4)
    C = mini_model.config
    kv_cache = paged_mod.PagedKVCache(
        C.num_layers, C.num_heads, C.head_dim, num_blocks=8, block_size=4
    )
    got = paged_generate_single(mini_model, prompt, n, pool, kv_cache)
    assert got == expected
