"""Week 7：LLM——给用户的那张"脸"。

对标 vLLM 的用法：

    llm = LLM(model="Qwen/Qwen3-0.6B")
    outs = llm.generate(["你好"], SamplingParams(max_new_tokens=32))

LLM 负责把"模型名字"变成一台组装好的 LLMEngine：
下载/定位权重、加载模型、建分词器、配好 KV cache 容量。
"""

import torch

from ..model.qwen3 import Qwen3ForCausalLM
from ..sampling.sampler import SamplingParams
from .llm_engine import LLMEngine


def _best_device():
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # _best_device


class LLM:
    """mini 版 vllm.LLM：一行起引擎，一行生成。"""

    def __init__(
        self,
        model_name="Qwen/Qwen3-0.6B",
        max_model_len=2048,
        device=None,
        dtype=None,
        num_blocks=None,
        block_size=16,
        max_num_seqs=8,
    ):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        self.device = device or _best_device()
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        self.dtype = dtype

        # 命中本地 HF 缓存；allow_patterns 避开缓存里可能缺失的 LICENSE 等无关文件
        model_path = snapshot_download(
            model_name,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path, device=self.device, dtype=dtype
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # KV cache 容量：默认装得下 max_num_seqs 条 max_model_len 长的序列
        if num_blocks is None:
            total_tokens = max_model_len * max_num_seqs
            num_blocks = -(-total_tokens // block_size)  # 向上取整
        self.engine = LLMEngine(
            self.model,
            self.tokenizer,
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _tokenize(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return [self.tokenizer.encode(p) for p in prompts]

    def generate(self, prompts, sampling_params=None):
        """批量生成。prompts 是字符串或字符串列表；返回 list[RequestOutput]。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # LLM.generate

    def stream(self, prompt, sampling_params=None):
        """流式生成单个 prompt，逐步吐出文本增量。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # LLM.stream
