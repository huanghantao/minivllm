"""Week 5：调度器与 continuous batching。"""

import torch

from tests._impl import load
from tests.test_w4 import paged_generate_single

pool_mod = load("cache.block_pool")
paged_mod = load("cache.paged_cache")
req_mod = load("scheduler.request")
sched_mod = load("scheduler.scheduler")
batching = load("scheduler.batching")
sampler_mod = load("sampling.sampler")


def make_request(rid, prompt, max_new_tokens=4):
    params = sampler_mod.SamplingParams(max_new_tokens=max_new_tokens)
    return req_mod.Request(rid, prompt, params)


# ---------- Request ----------

def test_request_basics():
    req = make_request(0, [10, 20, 30])
    assert req.num_prompt_tokens == 3
    assert req.num_tokens == 3
    assert req.last_token_id() == 30
    assert not req.is_finished()
    req.output_token_ids.append(99)
    assert req.last_token_id() == 99
    assert req.num_tokens == 4


# ---------- Scheduler ----------

def test_scheduler_fifo_admission():
    pool = pool_mod.BlockPool(num_blocks=64, block_size=4)
    sch = sched_mod.Scheduler(pool, max_num_seqs=2)
    for i in range(3):
        sch.add_request(make_request(i, [1, 2, 3]))
    out = sch.schedule()
    # max_num_seqs=2：只准入前两个，第三个继续排队
    assert [r.request_id for r in out.prefill] == [0, 1]
    assert out.decode == []
    assert len(sch.waiting) == 1
    # 下一步：两个在跑的进入 decode
    out = sch.schedule()
    assert out.prefill == []
    assert [r.request_id for r in out.decode] == [0, 1]


def test_scheduler_conservative_admission():
    """块不够装下"提示词+全部待生成"时，不许进门。"""
    pool = pool_mod.BlockPool(num_blocks=2, block_size=4)  # 共 8 格
    sch = sched_mod.Scheduler(pool, max_num_seqs=4)
    sch.add_request(make_request(0, [1, 2, 3], max_new_tokens=5))   # 3+5=8 格，正好
    sch.add_request(make_request(1, [1, 2, 3], max_new_tokens=6))   # 3+6=9 格，不够
    out = sch.schedule()
    assert [r.request_id for r in out.prefill] == [0]
    assert len(sch.waiting) == 1


def test_scheduler_finish_frees_blocks():
    pool = pool_mod.BlockPool(num_blocks=4, block_size=4)
    sch = sched_mod.Scheduler(pool, max_num_seqs=2)
    sch.add_request(make_request(0, [1, 2, 3], max_new_tokens=2))
    out = sch.schedule()
    bt = out.prefill[0].block_table
    for _ in range(3):
        bt.append_slot()
    free_before = pool.num_free_blocks()
    # 第 1 步：生成 1 个，还没够
    done = sch.update_after_step({0: 50})
    assert done == []
    # 第 2 步：够 2 个了，以 "length" 结束，块全部归还
    done = sch.update_after_step({0: 51})
    assert [r.request_id for r in done] == [0]
    assert done[0].finish_reason == "length"
    assert done[0].output_token_ids == [50, 51]
    assert pool.num_free_blocks() == free_before + 1  # 块 0 回来了
    assert not sch.has_unfinished()


def test_scheduler_stop_token():
    pool = pool_mod.BlockPool(num_blocks=4, block_size=4)
    sch = sched_mod.Scheduler(pool, max_num_seqs=2)
    sch.add_request(make_request(0, [1, 2, 3], max_new_tokens=10))
    sch.schedule()
    done = sch.update_after_step({0: 7}, eos_token_ids={7})
    assert done[0].finish_reason == "stop"


# ---------- continuous_batch_generate ----------

def make_kv_cache_for(model, num_blocks=16, block_size=4):
    C = model.config
    return paged_mod.PagedKVCache(
        C.num_layers, C.num_heads, C.head_dim, num_blocks, block_size
    )


def test_continuous_batching_matches_single(mini_model):
    """批处理绝不能改变贪心结果：每个请求的答案必须和单独跑一致。"""
    prompts = [[3, 1, 4], [1, 5, 9, 2, 6], [2, 7]]
    n = 5
    batched = batching.continuous_batch_generate(
        mini_model, prompts, max_new_tokens=n,
        num_blocks=16, block_size=4, max_num_seqs=3,
    )
    assert len(batched) == 3
    for prompt, expect in zip(prompts, batched):
        pool = pool_mod.BlockPool(num_blocks=16, block_size=4)
        kv = make_kv_cache_for(mini_model)
        single = paged_generate_single(mini_model, prompt, n, pool, kv)
        assert expect == single


def test_continuous_batching_block_reuse(mini_model):
    """池子只够单条请求时，批处理应当排队串行跑完——结果仍然正确。"""
    prompts = [[3, 1, 4], [1, 5, 9]]
    n = 4
    # 3 + 4 = 7 token，块大小 4 → 2 块；池子只给 2 块
    out = batching.continuous_batch_generate(
        mini_model, prompts, max_new_tokens=n,
        num_blocks=2, block_size=4, max_num_seqs=8,
    )
    assert len(out) == 2
    for prompt, expect in zip(prompts, out):
        pool = pool_mod.BlockPool(num_blocks=2, block_size=4)
        kv = make_kv_cache_for(mini_model, num_blocks=2)
        single = paged_generate_single(mini_model, prompt, n, pool, kv)
        assert expect == single
