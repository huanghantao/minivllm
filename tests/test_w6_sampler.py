"""Week 6（前半）：采样器。"""

import torch

from tests._impl import load

sampler_mod = load("sampling.sampler")
SamplingParams = sampler_mod.SamplingParams


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 2.3, -1.0, 2.29])
    assert sampler_mod.greedy_sample(logits) == 1


def test_apply_temperature():
    logits = torch.tensor([1.0, 2.0])
    assert torch.equal(sampler_mod.apply_temperature(logits, 2.0), torch.tensor([0.5, 1.0]))


def test_top_k_filter():
    logits = torch.tensor([1.0, 5.0, 2.0, 4.0])
    out = sampler_mod.top_k_filter(logits, 2)
    assert out[1].item() == 5.0 and out[3].item() == 4.0
    assert out[0].item() == float("-inf") and out[2].item() == float("-inf")
    # k 超过词表或 <= 0：不动
    assert torch.equal(sampler_mod.top_k_filter(logits, -1), logits)
    assert torch.equal(sampler_mod.top_k_filter(logits, 99), logits)


def test_top_p_filter():
    # 概率 [0.5, 0.3, 0.15, 0.05]（按分数排好序后的相对位置）
    logits = torch.log(torch.tensor([0.05, 0.5, 0.15, 0.3]))
    out = sampler_mod.top_p_filter(logits, 0.6)
    # 保留 0.5 和 0.3（累计 0.8 刚超过 0.6）；0.15 之前累计已达 0.8 → 去掉
    kept = torch.isfinite(out).nonzero().flatten().tolist()
    assert kept == [1, 3]


def test_top_p_keeps_at_least_one():
    logits = torch.tensor([10.0, 0.0, 0.0])
    out = sampler_mod.top_p_filter(logits, 0.01)
    assert torch.isfinite(out).sum().item() == 1
    assert torch.isfinite(out)[0]


def test_sample_token_greedy_ignores_filters():
    logits = torch.tensor([1.0, 5.0, 2.0])
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.01)
    assert sampler_mod.sample_token(logits, params) == 1


def test_sample_token_reproducible_with_seed():
    torch.manual_seed(0)
    logits = torch.randn(50)
    params = SamplingParams(temperature=1.0)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = [sampler_mod.sample_token(logits, params, g1) for _ in range(5)]
    b = [sampler_mod.sample_token(logits, params, g2) for _ in range(5)]
    assert a == b


def test_sample_token_respects_top_k_support():
    torch.manual_seed(0)
    logits = torch.zeros(10)
    logits[3] = 100.0
    logits[7] = 99.0
    params = SamplingParams(temperature=1.0, top_k=2)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert sampler_mod.sample_token(logits, params, g) in (3, 7)


def test_sampler_per_request_seed():
    s = sampler_mod.Sampler()
    logits = torch.randn(50)
    params = SamplingParams(temperature=1.0, seed=7)
    a = s.sample_one(logits, request_id=1, params=params)
    s.drop_request(1)
    b = s.sample_one(logits, request_id=1, params=params)  # 重建 generator → 重来
    assert a == b
