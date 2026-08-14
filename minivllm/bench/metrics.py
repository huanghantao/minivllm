"""Week 8：metrics——给引擎量体温。

学推理优化，不看数字等于白学。两个最重要的指标：
- TTFT（Time To First Token）：首 token 延迟——用户多久看到第一个字；
- 吞吐（throughput）：每秒生成多少 token——引擎一秒能服务多少输出。

这个文件故意写成"函数接受 callable"的形式：
测试里可以塞假函数进来，不用真跑模型。
"""

import time
from dataclasses import dataclass


@dataclass
class BenchResult:
    """一次 benchmark 的成绩单。"""

    num_prompts: int
    num_output_tokens: int
    total_time: float          # 整批跑完的总秒数
    ttft: float | None = None  # 首 token 延迟（流式单请求时测）

    @property
    def tokens_per_second(self):
        """吞吐：总输出 token 数 / 总时间。"""
        if self.total_time <= 0:
            return float("inf")
        return self.num_output_tokens / self.total_time


def benchmark_throughput(generate_fn, prompts, sampling_params=None):
    """测吞吐：跑一批 prompt，统计总输出 token 数和总耗时。

    generate_fn: 一个 callable(prompts, sampling_params) -> outputs，
                 outputs 里每个元素要有 .output_token_ids。
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # benchmark_throughput


def measure_ttft(stream_fn):
    """测 TTFT：流式生成时，从发起到收到第一片文本的秒数。

    stream_fn: 一个 callable() -> iterator（每次 yield 一片文本）。
    返回 (ttft 秒数, 完整文本)。
    """
    raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # measure_ttft


def format_report(result):
    """把成绩单排版成好读的字符串。"""
    lines = [
        f"请求数:        {result.num_prompts}",
        f"输出 token 数: {result.num_output_tokens}",
        f"总耗时:        {result.total_time:.2f} s",
        f"吞吐:          {result.tokens_per_second:.1f} tok/s",
    ]
    if result.ttft is not None:
        lines.append(f"TTFT:          {result.ttft * 1000:.0f} ms")
    return "\n".join(lines)
