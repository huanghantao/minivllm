"""实测：naive（无 cache 全量重算）vs HF 自带 cache 的生成速度。

用 minivllm 项目自己的 .venv 跑：
    .venv/bin/python scripts/bench_naive_hf.py
结果合并进 figures/data/bench.json，供生图脚本使用。
"""

import json
import pathlib
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data" / "bench.json"

PROMPT = "The capital of France is"
MAX_NEW = 64


def measure(generate_fn):
    # 预热一次再测
    generate_fn()
    start = time.perf_counter()
    generate_fn()
    return MAX_NEW / (time.perf_counter() - start)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = tok.encode(PROMPT, return_tensors="pt").to(device)

    @torch.no_grad()
    def naive():
        # 每步整段重算（use_cache=False）——Week 1 的写法
        cur = ids
        for _ in range(MAX_NEW):
            out = model(cur, use_cache=False)
            nxt = out.logits[0, -1].argmax()
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)

    @torch.no_grad()
    def cached():
        model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=0)

    results = {
        "naive_hf_tps": round(measure(naive), 1),
        "cached_hf_tps": round(measure(cached), 1),
    }
    print(results)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(DATA.read_text()) if DATA.exists() else {}
    old.update(results)
    DATA.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    print(f"merged into {DATA}")


if __name__ == "__main__":
    main()
