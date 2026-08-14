"""Week 6（后半）：Qwen3 真实模型——和 HuggingFace 对拍。

这些测试要加载真实的 Qwen3-0.6B 权重（约 1.2GB），标记为 slow。
跑法：IMPL=reference pytest tests/test_w6_qwen3.py -m slow
"""

import os

import pytest
import torch

from tests._impl import load

qwen3_mod = load("model.qwen3")
generate_mod = load("generate")
pool_mod = load("cache.block_pool")
table_mod = load("cache.block_table")
paged_mod = load("cache.paged_cache")

MODEL_NAME = "Qwen/Qwen3-0.6B"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model_path():
    from huggingface_hub import snapshot_download
    # 只要代码需要的文件，避开缓存里缺失的 LICENSE/README 等
    return snapshot_download(
        MODEL_NAME, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]
    )


@pytest.fixture(scope="module")
def our_model(model_path):
    model = qwen3_mod.Qwen3ForCausalLM.from_pretrained(
        model_path, device="cpu", dtype=torch.float32
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_model(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def prompt_ids(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    return torch.tensor(tok.encode("The capital of France is"))


def test_logits_match_hf(our_model, hf_model, prompt_ids):
    """同样的输入，我们的 logits 必须和 HF 的几乎一致。"""
    with torch.no_grad():
        ours = our_model(prompt_ids.unsqueeze(0))[0]
        theirs = hf_model(prompt_ids.unsqueeze(0)).logits[0]
    diff = (ours - theirs).abs().max().item()
    assert diff < 5e-3, f"logits 最大偏差 {diff}"


def test_greedy_continuation_matches_hf(our_model, hf_model, prompt_ids):
    """贪心生成 8 个词，两边必须一模一样。"""
    with torch.no_grad():
        ours = generate_mod.generate_naive(
            our_model, prompt_ids.unsqueeze(0), max_new_tokens=8
        )[0, prompt_ids.shape[0]:].tolist()
        out = hf_model.generate(
            prompt_ids.unsqueeze(0), max_new_tokens=8, do_sample=False,
            pad_token_id=0,
        )
        theirs = out[0, prompt_ids.shape[0]:].tolist()
    assert ours == theirs


def test_paged_path_matches_naive(our_model, prompt_ids):
    """Qwen3 的分页引擎接口（prefill + decode_batch）== 朴素整段前向。"""
    C = our_model.config
    pool = pool_mod.BlockPool(num_blocks=64, block_size=16)
    kv = paged_mod.PagedKVCache(
        C.num_layers, C.num_kv_heads, C.head_dim, num_blocks=64, block_size=16
    )
    bt = table_mod.BlockTable(pool)
    ids = prompt_ids.tolist()
    slots = [bt.append_slot() for _ in ids]
    logits = our_model.prefill(prompt_ids, kv, slots, bt)
    # prefill 的最后一个 logits 应等于朴素前向的最后一个
    with torch.no_grad():
        ref = our_model(prompt_ids.unsqueeze(0))[0, -1]
    assert torch.allclose(logits, ref, atol=1e-4)

    # decode 两步，逐步与朴素前向对拍
    next_ids = [int(torch.argmax(logits).item())]
    for _ in range(2):
        slot = bt.append_slot()
        token = torch.tensor([next_ids[-1]])
        pos = torch.tensor([bt.num_tokens - 1])
        logits = our_model.decode_batch(token, pos, kv, [slot], [bt])[0]
        next_ids.append(int(torch.argmax(logits).item()))
    full = torch.tensor(ids + next_ids[:-1])
    with torch.no_grad():
        ref_logits = our_model(full.unsqueeze(0))[0]
    for i, tok in enumerate(next_ids):
        step_ref = int(torch.argmax(ref_logits[len(ids) - 1 + i]).item())
        assert tok == step_ref
