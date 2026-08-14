"""Week 6：Sampler——从 logits 到"下一个词"的最后一公里。

模型吐出来的是 vocab 个分数（logits），怎么从分数变成词？
- greedy：永远挑最高的（deterministic，可复现）；
- temperature：调节"胆子"大小——越小越保守，越大越野；
- top_k：只在分数最高的 k 个里挑；
- top_p：只保留累计概率到 p 的那一小撮（动态个数）。

这套流水线就是 vLLM SamplingParams 的 mini 版。
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    """一次生成请求的采样参数表。

    temperature = 0 表示贪心（greedy），此时 top_k / top_p 不生效。
    """

    temperature: float = 0.0
    top_k: int = -1          # -1 表示不截断
    top_p: float = 1.0       # 1.0 表示不截断
    max_new_tokens: int = 16
    stop_token_ids: tuple = field(default_factory=tuple)
    seed: int = None         # 给了 seed，采样结果可复现


def greedy_sample(logits):
    """贪心：挑分数最高的那个 token id。logits: (vocab,)"""
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # greedy_sample


def apply_temperature(logits, temperature):
    """温度缩放：logits / T。

    T < 1：分数差距被放大 → 更保守；T > 1：差距被抹平 → 更随机。
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # apply_temperature


def top_k_filter(logits, k):
    """只保留分数最高的 k 个，其余置 -inf（概率归零）。k <= 0 表示不截断。"""
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # top_k_filter


def top_p_filter(logits, p):
    """核采样（nucleus）：保留累计概率刚好盖过 p 的最小集合。

    做法：按分数从高到低排序、算累计概率，
    凡是"轮到它之前累计概率已经超过 p"的，置 -inf。
    至少保留第一名。p >= 1 表示不截断。
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # top_p_filter


def sample_token(logits, params=None, generator=None):
    """完整采样流水线：temperature → top_k → top_p → 按概率抽签。

    logits: (vocab,)；params 为 None 时等价贪心。
    params.temperature == 0 时走贪心（不看 top_k / top_p）。
    返回 token id（int）。
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # sample_token


class Sampler:
    """给引擎用的采样器：一次服务一批请求，各自有各自的参数。"""

    def __init__(self):
        # request_id -> torch.Generator（有 seed 的请求才有）
        self._generators = {}

    def _generator_for(self, request_id, seed, device):
        if seed is None:
            return None
        if request_id not in self._generators:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            self._generators[request_id] = g
        return self._generators[request_id]

    def sample_one(self, logits, request_id, params):
        """给单个请求采一个 token。logits: (vocab,)"""
        g = self._generator_for(request_id, params.seed, logits.device)
        if g is not None:
            logits = logits.cpu()
            token = sample_token(logits, params, generator=g)
        else:
            token = sample_token(logits, params)
        return token

    def drop_request(self, request_id):
        """请求结束后清掉它的随机数发生器。"""
        self._generators.pop(request_id, None)
