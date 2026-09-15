"""实测：真实 vLLM（v0.27.1，vllm-metal 后端）batch 1 / batch 8 的吞吐。

用课程的 .venv 跑（注意必须用脚本文件方式运行，vLLM 要多进程）：
    HF_HUB_OFFLINE=1 VLLM_METAL_USE_PAGED_ATTENTION=0 VLLM_HOST_IP=127.0.0.1 \
      .venv/bin/python scripts/bench_vllm.py
结果合并进 figures/data/bench.json。

VLLM_HOST_IP 不能省：代理工具开 TUN 模式时默认路由被虚拟网卡接管，
vLLM 探测本机 IP 会拿到 fake-IP 地址，内部 TCPStore 一连就断。

"""

import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def run_batch(llm, prompts):
    from vllm import SamplingParams
    start = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(max_tokens=MAX_NEW, temperature=0))
    dt = time.perf_counter() - start
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    return n / dt


def main():
    from vllm import LLM

    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=512)
    run_batch(llm, PROMPTS[:1])  # 预热
    results = {
        "vllm_b1_tps": round(run_batch(llm, PROMPTS[:1]), 1),
        "vllm_b8_tps": round(run_batch(llm, PROMPTS), 1),
    }
    print(results)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(DATA.read_text()) if DATA.exists() else {}
    old.update(results)
    DATA.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    print(f"merged into {DATA}")


if __name__ == "__main__":
    main()
