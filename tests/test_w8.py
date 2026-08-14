"""Week 8：性能测量工具。"""

import time

from tests._impl import load

metrics = load("bench.metrics")


class FakeOutput:
    def __init__(self, n):
        self.output_token_ids = list(range(n))


def test_bench_result_throughput():
    r = metrics.BenchResult(num_prompts=2, num_output_tokens=50, total_time=2.0)
    assert r.tokens_per_second == 25.0


def test_benchmark_throughput_aggregates():
    def fake_generate(prompts, params):
        time.sleep(0.01)
        return [FakeOutput(10) for _ in prompts]

    result = metrics.benchmark_throughput(fake_generate, ["a", "b", "c"])
    assert result.num_prompts == 3
    assert result.num_output_tokens == 30
    assert result.total_time >= 0.01
    assert result.tokens_per_second > 0


def test_measure_ttft_and_text():
    def fake_stream():
        time.sleep(0.01)
        yield "你"
        yield "好"
        yield "吗"

    ttft, text = metrics.measure_ttft(fake_stream)
    assert 0.01 <= ttft < 1.0
    assert text == "你好吗"


def test_format_report():
    r = metrics.BenchResult(
        num_prompts=2, num_output_tokens=50, total_time=2.0, ttft=0.123
    )
    report = metrics.format_report(r)
    assert "25.0 tok/s" in report
    assert "123 ms" in report
