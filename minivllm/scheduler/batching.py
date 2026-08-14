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
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # continuous_batch_generate
