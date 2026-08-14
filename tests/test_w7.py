"""Week 7：LLMEngine 与 LLM——组装、批量、流式。"""

import pytest
import torch

from tests._impl import load
from tests.conftest import ToyTokenizer
from tests.test_w4 import paged_generate_single

engine_mod = load("engine.llm_engine")
sampler_mod = load("sampling.sampler")
pool_mod = load("cache.block_pool")
paged_mod = load("cache.paged_cache")

SamplingParams = sampler_mod.SamplingParams


@pytest.fixture
def engine(mini_model):
    corpus = ["a b c", "d e f", "a d g"]
    tok = ToyTokenizer(corpus)
    # mini_model 词表 64，玩具分词器词表只有几个词，足够
    eng = engine_mod.LLMEngine(
        mini_model, tok, num_blocks=16, block_size=4, max_num_seqs=2
    )
    return eng, tok


def test_engine_generate_matches_single(engine, mini_model):
    eng, tok = engine
    prompt = tok.encode("a b c")
    params = SamplingParams(max_new_tokens=5)
    outs = eng.generate([prompt], params)
    assert len(outs) == 1
    assert outs[0].finish_reason == "length"

    pool = pool_mod.BlockPool(num_blocks=16, block_size=4)
    C = mini_model.config
    kv = paged_mod.PagedKVCache(C.num_layers, C.num_heads, C.head_dim, 16, 4)
    expect = paged_generate_single(mini_model, prompt, 5, pool, kv)
    assert outs[0].output_token_ids == expect


def test_engine_batch_order_and_consistency(engine):
    """批量跑的结果 = 各自单独跑的结果，且按提交顺序返回。"""
    eng, tok = engine
    prompts = [tok.encode(t) for t in ["a b c", "d e f", "a d g"]]
    params = SamplingParams(max_new_tokens=4)
    batched = eng.generate(prompts, params)
    assert [o.request_id for o in batched] == [0, 1, 2]
    for out, p in zip(batched, prompts):
        single = eng.generate([p], params)[0]
        assert out.output_token_ids == single.output_token_ids


def test_engine_stream_concat_equals_generate(engine):
    eng, tok = engine
    prompt = tok.encode("a b c")
    params = SamplingParams(max_new_tokens=5)
    pieces = list(eng.stream(prompt, params))
    text_from_stream = "".join(pieces)
    out = eng.generate([tok.encode("a b c")], params)[0]
    assert text_from_stream == out.text


def test_engine_stop_token(engine, mini_model):
    """先探出第一个生成词，再把它设为 stop，应当只生成一个词。"""
    eng, tok = engine
    prompt = tok.encode("a b c")
    probe = eng.generate([prompt], SamplingParams(max_new_tokens=1))[0]
    first_token = probe.output_token_ids[0]
    assert probe.finish_reason == "length"

    params = SamplingParams(max_new_tokens=10, stop_token_ids=(first_token,))
    out = eng.generate([prompt], params)[0]
    assert out.output_token_ids == [first_token]
    assert out.finish_reason == "stop"


# ---------- 真实模型端到端（慢）----------

@pytest.mark.slow
def test_llm_facade_real_model():
    llm_mod = load("engine.llm")
    llm = llm_mod.LLM(
        model_name="Qwen/Qwen3-0.6B", max_model_len=512, max_num_seqs=4
    )
    outs = llm.generate(
        ["The capital of France is", "1+1="],
        SamplingParams(max_new_tokens=16),
    )
    assert "Paris" in outs[0].text
    assert "2" in outs[1].text

    pieces = list(llm.stream("The capital of France is", SamplingParams(max_new_tokens=16)))
    assert "Paris" in "".join(pieces)
