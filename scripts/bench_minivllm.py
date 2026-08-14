"""实测：我们自己的 minivllm 引擎（reference）batch 1 / batch 8 吞吐 + TTFT。

用 minivllm 项目自己的 .venv 跑：
    .venv/bin/python scripts/bench_minivllm.py
结果合并进 figures/data/bench.json。
"""

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "figures" / "data" / "bench.json"

PROMPTS = [
    "The capital of France is",
    "Explain what a KV cache is in one sentence:",
    "1+1=",
    "Write a haiku about memory:",
    "The quick brown fox",
    "List three colors:",
    "What is paging in operating systems?",
    "Translate 'hello' to French:",
]
MAX_NEW = 64


def main():
    from reference.bench.metrics import measure_ttft
    from reference.engine.llm import LLM
    from reference.sampling.sampler import SamplingParams

    llm = LLM(model_name="Qwen/Qwen3-0.6B", max_model_len=512, max_num_seqs=8)
    params = SamplingParams(max_new_tokens=MAX_NEW)

    def run(prompts):
        start = time.perf_counter()
        outs = llm.generate(prompts, params)
        dt = time.perf_counter() - start
        n = sum(len(o.output_token_ids) for o in outs)
        return n / dt

    run(PROMPTS[:1])  # 预热
    ttft, _ = measure_ttft(lambda: llm.stream(PROMPTS[0], params))
    results = {
        "minivllm_b1_tps": round(run(PROMPTS[:1]), 1),
        "minivllm_b8_tps": round(run(PROMPTS), 1),
        "minivllm_ttft_ms": round(ttft * 1000),
    }
    print(results)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(DATA.read_text()) if DATA.exists() else {}
    old.update(results)
    DATA.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    print(f"merged into {DATA}")


if __name__ == "__main__":
    main()
