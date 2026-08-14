"""Week 5：continuous_batch_generate—— proto-引擎（引擎的雏形）。

把前面几周的零件第一次全部组装起来跑批：
分页 KV cache（W4）+ 调度器（W5）+ 模型引擎接口（W4/W5）。

这是一个函数，不是类——Week 7 会把它重构成 LLMEngine。
它默认贪心采样（采样器 Week 6 才来）。
"""

import torch

from ..cache.block_pool import BlockPool
from ..cache.paged_cache import PagedKVCache
from ..sampling.sampler import SamplingParams
from .request import Request
from .scheduler import Scheduler


@torch.no_grad()
def continuous_batch_generate(
    model,
    prompts,
    max_new_tokens=16,
    num_blocks=64,
    block_size=4,
    max_num_seqs=8,
    eos_token_ids=(),
):
    """对一批提示词做 continuous batching 生成。

    参数：
        model: 实现了引擎接口（prefill / decode_batch）的模型。
        prompts: list[list[int]]，每个请求的提示词 token id。
        max_new_tokens: 每个请求最多生成多少词（统一）。
        num_blocks / block_size: 分页 KV cache 的容量。
        max_num_seqs: 批里最多同时跑几个请求。
        eos_token_ids: 结束符集合。

    返回：
        list[list[int]]：与 prompts 同序的生成结果（不含提示词）。
    """
    device = next(model.parameters()).device
    C = model.config
    pool = BlockPool(num_blocks, block_size)
    kv_cache = PagedKVCache(
        num_layers=C.num_layers,
        num_heads=C.num_heads,
        head_dim=C.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
    )
    scheduler = Scheduler(pool, max_num_seqs=max_num_seqs)

    requests = []
    for i, prompt in enumerate(prompts):
        params = SamplingParams(max_new_tokens=max_new_tokens)
        req = Request(i, prompt, params)
        scheduler.add_request(req)
        requests.append(req)

    while scheduler.has_unfinished():
        out = scheduler.schedule()
        sampled = {}

        # ---- prefill：新准入的请求，逐个跑完整提示词 ----
        for req in out.prefill:
            slots = [
                req.block_table.append_slot()
                for _ in range(req.num_prompt_tokens)
            ]
            ids = torch.tensor(req.prompt_token_ids, dtype=torch.long, device=device)
            logits = model.prefill(ids, kv_cache, slots, req.block_table)
            sampled[req.request_id] = int(torch.argmax(logits).item())

        # ---- decode：在跑的请求，批量各走一步 ----
        if out.decode:
            slots = [req.block_table.append_slot() for req in out.decode]
            token_ids = torch.tensor(
                [req.last_token_id() for req in out.decode],
                dtype=torch.long, device=device,
            )
            positions = torch.tensor(
                [req.block_table.num_tokens - 1 for req in out.decode],
                dtype=torch.long, device=device,
            )
            logits = model.decode_batch(
                token_ids, positions, kv_cache, slots,
                [req.block_table for req in out.decode],
            )
            for i, req in enumerate(out.decode):
                sampled[req.request_id] = int(torch.argmax(logits[i]).item())

        scheduler.update_after_step(sampled, eos_token_ids)

    return [req.output_token_ids for req in requests]
