"""Week 2：注意力数学 + MiniTransformer。"""

import torch

from tests._impl import load

attention = load("model.attention")


# ---------- make_causal_mask ----------

def test_causal_mask_square():
    mask = attention.make_causal_mask(3)
    expected = torch.tensor(
        [[True, False, False],
         [True, True, False],
         [True, True, True]]
    )
    assert torch.equal(mask, expected)


def test_causal_mask_with_cache():
    # 2 个新词 + 2 个旧词：旧词全能看，新词之间遵守因果
    mask = attention.make_causal_mask(2, 4)
    expected = torch.tensor(
        [[True, True, True, False],
         [True, True, True, True]]
    )
    assert torch.equal(mask, expected)


# ---------- scaled_dot_product_attention ----------

def test_sdpa_hand_computed():
    # q = [1, 0]，两个键 k1 = [1, 0]、k2 = [0, 1]
    # 分数 = [1/sqrt(2), 0]，权重 = softmax([0.7071, 0])
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
    out, weights = attention.scaled_dot_product_attention(q, k, v)
    import math
    expected_w = torch.softmax(torch.tensor([1 / math.sqrt(2), 0.0]), dim=-1)
    assert torch.allclose(weights[0, 0, 0], expected_w, atol=1e-6)
    expected_out = expected_w[0] * v[0, 0, 0] + expected_w[1] * v[0, 0, 1]
    assert torch.allclose(out[0, 0, 0], expected_out, atol=1e-6)


def test_sdpa_weights_sum_to_one():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 8)
    _, weights = attention.scaled_dot_product_attention(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_sdpa_mask_blocks_future():
    q = torch.ones(1, 1, 2, 4)
    k = torch.ones(1, 1, 2, 4)
    v = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]]])
    mask = attention.make_causal_mask(2)
    _, weights = attention.scaled_dot_product_attention(q, k, v, mask)
    # 第 0 个查询看不到第 1 个键（未来）
    assert weights[0, 0, 0, 1].item() == 0.0
    # 第 1 个查询两个都能看到
    assert weights[0, 0, 1, 1].item() > 0.0


# ---------- MiniTransformer ----------

def test_transformer_output_shape(mini_model):
    ids = torch.tensor([[1, 2, 3]])
    logits = mini_model(ids)
    assert logits.shape == (1, 3, mini_model.config.vocab_size)


def test_transformer_deterministic(mini_model):
    ids = torch.tensor([[5, 6, 7, 8]])
    assert torch.equal(mini_model(ids), mini_model(ids))


def test_transformer_causality(mini_model):
    """改未来的 token，过去位置的预测不能变。"""
    ids1 = torch.tensor([[1, 2, 3, 4]])
    ids2 = torch.tensor([[1, 2, 9, 9]])  # 前两个位置相同，后两个改了
    l1 = mini_model(ids1)
    l2 = mini_model(ids2)
    assert torch.allclose(l1[:, :2], l2[:, :2], atol=1e-6)
