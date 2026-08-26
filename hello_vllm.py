"""第一次和真实 vLLM 打招呼。

运行（必须写成带 main 守卫的 .py 脚本文件，且必须加两个环境变量）：
    HF_HUB_OFFLINE=1 VLLM_METAL_USE_PAGED_ATTENTION=0 \
      .venv/bin/python hello_vllm.py
"""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=512)

    outs = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=64, temperature=0),
    )
    print(outs[0].outputs[0].text)


if __name__ == "__main__":
    main()
