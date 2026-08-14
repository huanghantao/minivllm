"""Week 7：LLMEngine——把零件组装成一台引擎。

Week 5 的 continuous_batch_generate 是一个"一次性函数"：
请求一次给齐、结果一次拿全。

真实引擎（vLLM 的 LLMEngine）是长跑的：
- add_request：请求随时可以进来；
- step：引擎走一步（一轮点名 + 一轮计算）；
- generate：同步地"进一批、跑到完、一起拿"；
- stream：流式地一个一个 token 往外吐。

本文件就是把 Week 5 那个函数拆成一台可以长跑的机器。
"""

import torch

from ..cache.block_pool import BlockPool
from ..cache.paged_cache import PagedKVCache
from ..sampling.sampler import Sampler, SamplingParams
from ..scheduler.request import Request
from ..scheduler.scheduler import Scheduler


class RequestOutput:
    """一次生成的最终结果（对标 vLLM 的 RequestOutput 的极简版）。"""

    def __init__(self, request, text):
        self.request_id = request.request_id
        self.prompt_token_ids = request.prompt_token_ids
        self.output_token_ids = request.output_token_ids
        self.text = text
        self.finish_reason = request.finish_reason

    def __repr__(self):
        return (
            f"RequestOutput(id={self.request_id}, "
            f"finish={self.finish_reason}, text={self.text!r})"
        )


class LLMEngine:
    """mini 版 LLM 引擎：调度器 + 分页缓存 + 采样器 + 模型。"""

    def __init__(
        self,
        model,
        tokenizer,
        num_blocks=256,
        block_size=16,
        max_num_seqs=8,
        eos_token_id=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if eos_token_id is None:
            self.eos_token_ids = ()
        elif isinstance(eos_token_id, int):
            self.eos_token_ids = (eos_token_id,)
        else:
            self.eos_token_ids = tuple(eos_token_id)
        self.device = next(model.parameters()).device
        C = model.config
        self.block_pool = BlockPool(num_blocks, block_size)
        self.kv_cache = PagedKVCache(
            num_layers=C.num_layers,
            num_heads=C.num_kv_heads if hasattr(C, "num_kv_heads") else C.num_heads,
            head_dim=C.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            device=self.device,
            dtype=next(model.parameters()).dtype,
        )
        self.scheduler = Scheduler(self.block_pool, max_num_seqs=max_num_seqs)
        self.sampler = Sampler()
        self._next_request_id = 0
        self._requests = {}  # request_id -> Request（含已完成的）

    # ---- 请求的进与出 ----

    def add_request(self, prompt_token_ids, sampling_params=None):
        """提交一个请求（token id 序列），返回 request_id。"""
        params = sampling_params or SamplingParams()
        req = Request(self._next_request_id, prompt_token_ids, params)
        self._next_request_id += 1
        self.scheduler.add_request(req)
        self._requests[req.request_id] = req
        return req.request_id

    def has_unfinished(self):
        return self.scheduler.has_unfinished()

    # ---- 引擎的心跳 ----

    @torch.no_grad()
    def step(self):
        """走一步：点名 → prefill 新请求 → 批量 decode → 登记结果。

        返回本步刚完成的 request_id 列表。
        """
        out = self.scheduler.schedule()
        sampled = {}

        for req in out.prefill:
            slots = [
                req.block_table.append_slot()
                for _ in range(req.num_prompt_tokens)
            ]
            ids = torch.tensor(
                req.prompt_token_ids, dtype=torch.long, device=self.device
            )
            logits = self.model.prefill(ids, self.kv_cache, slots, req.block_table)
            token = self.sampler.sample_one(
                logits, req.request_id, req.sampling_params
            )
            sampled[req.request_id] = token

        if out.decode:
            slots = [req.block_table.append_slot() for req in out.decode]
            token_ids = torch.tensor(
                [req.last_token_id() for req in out.decode],
                dtype=torch.long, device=self.device,
            )
            positions = torch.tensor(
                [req.block_table.num_tokens - 1 for req in out.decode],
                dtype=torch.long, device=self.device,
            )
            logits = self.model.decode_batch(
                token_ids, positions, self.kv_cache, slots,
                [req.block_table for req in out.decode],
            )
            for i, req in enumerate(out.decode):
                token = self.sampler.sample_one(
                    logits[i], req.request_id, req.sampling_params
                )
                sampled[req.request_id] = token

        finished = self.scheduler.update_after_step(sampled, self.eos_token_ids)
        for req in finished:
            self.sampler.drop_request(req.request_id)
        return [req.request_id for req in finished]

    # ---- 两种用法 ----

    def generate(self, prompts_token_ids, sampling_params=None):
        """同步生成：进一批、跑到完、按提交顺序返回 list[RequestOutput]。"""
        ids = [
            self.add_request(p, sampling_params) for p in prompts_token_ids
        ]
        while self.has_unfinished():
            self.step()
        return [self._make_output(self._requests[i]) for i in ids]

    def stream(self, prompt_token_ids, sampling_params=None):
        """流式生成（单请求）：每走一步，吐出新生成的文本增量。"""
        rid = self.add_request(prompt_token_ids, sampling_params)
        req = self._requests[rid]
        sent = 0
        while True:
            finished = self.step()
            text = self.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            if len(text) > sent:
                yield text[sent:]
                sent = len(text)
            if rid in finished:
                break

    def _make_output(self, req):
        text = self.tokenizer.decode(
            req.output_token_ids, skip_special_tokens=True
        )
        return RequestOutput(req, text)
