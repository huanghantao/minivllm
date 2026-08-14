# 第 4 章：性能测量——TTFT 与吞吐

> 本章你将：
> 1. 搞清楚推理引擎的两个核心指标：**TTFT**（首字延迟）和**吞吐**（每秒 token 数）各管什么；
> 2. 亲手实现 `benchmark_throughput` 和 `measure_ttft`——给引擎量体温的两支体温计；
> 3. 学会测量的一条铁律：**先预热，再计时**；
> 4. 跑绿 `pytest tests/test_w8.py`，为第 5 章的结业实测备好工具。

---

## 4.1 学优化不看数字，等于白学

前 7 周我们一直在说"这个优化快""那个优化省"，但快多少、省多少，
一直靠别人的数字（生图脚本背后的实测）。这一周我们自己造测量工具——
因为**性能优化这行有个铁律：没有数字的优化都是玄学。**

先认识两个最重要的指标。想象你在用聊天机器人：

- 你按下回车，**多久看到第一个字**？这叫 **TTFT（Time To First Token，
  首 token 延迟）**。它决定用户觉得"这 AI 反应快不快"。TTFT 主要由
  prefill 决定：提示词越长、队列越挤，第一个字来得越慢；
- 第一个字出现之后，**后面的话流得多快**？这叫**吞吐（throughput）**，
  单位是 token/秒。单个请求时它叫"解码速度"，一批请求一起跑时它是
  "整个引擎一秒能服务多少输出"—— Week 5 我们就是把 8 个请求的吞吐
  从 19.6 抬到了 62.3 tok/s。

一个管"第一下的体验"，一个管"持续的能力"。优化 TTFT 的手段（chunked prefill、
prefix caching）和优化吞吐的手段（continuous batching、分页）不完全相同——
所以两个都要测。

## 4.2 成绩单：BenchResult（已给好）

打开 `minivllm/bench/metrics.py`。这个文件故意写成"函数接受 callable"的形式——
测试里可以塞假函数进来，不用真跑模型，所以 Week 8 的测试是**快测**，不加载权重。

文件开头是一张"成绩单"数据类（已给好，读一遍）：

```python
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
```

`@dataclass` 是 Python 标准库的装饰器：自动帮你生成 `__init__`，
四个字段直接变成构造参数。`tokens_per_second` 是个 `@property`——
调用时写 `result.tokens_per_second`（不加括号），读起来像个属性，
其实是现算的：**吞吐 = 总输出 token 数 ÷ 总秒数**，一句话的定义。

注意 `ttft` 默认是 `None`：整批压测时不测 TTFT（一批请求同时跑，
"第一个字"是谁的第一个字？），TTFT 只在**单请求流式**时有意义。
所以两个指标由两个函数分别测量。

## 4.3 实现 benchmark_throughput

这是你要填的第一个函数，合同在 docstring 里：

```python
def benchmark_throughput(generate_fn, prompts, sampling_params=None):
    """测吞吐：跑一批 prompt，统计总输出 token 数和总耗时。

    generate_fn: 一个 callable(prompts, sampling_params) -> outputs，
                 outputs 里每个元素要有 .output_token_ids。
    """
```

它不知道、也不关心 `generate_fn` 是我们的引擎、HF 还是真实 vLLM——
**只要是个"吃进 prompts 和参数、吐出带 output_token_ids 的结果列表"的
可调用对象就行。** 这正是它能通吃所有引擎的原因。

实现只有三步：

1. **开始计时**：`start = time.perf_counter()`。
   `perf_counter` 是 Python 里精度最高的计时器，专给测量用
   （别用 `time.time()`，那会受系统调表影响）；
2. **跑，然后停表**：

```python
   outputs = generate_fn(prompts, sampling_params)
   total = time.perf_counter() - start
```

3. **数 token、填成绩单**：`sum(len(o.output_token_ids) for o in outputs)`——
   生成式 `sum(...)` 把每个输出的 token 数加起来，塞进 `BenchResult` 返回。

就这么简单。测吞吐的全部智慧不在代码里，在**用法**里（见 4.5 的预热）。

## 4.4 实现 measure_ttft

第二个函数，合同：

```python
def measure_ttft(stream_fn):
    """测 TTFT：流式生成时，从发起到收到第一片文本的秒数。

    stream_fn: 一个 callable() -> iterator（每次 yield 一片文本）。
    返回 (ttft 秒数, 完整文本)。
    """
```

Week 7 你写过 `LLM.stream()`：一个迭代器，逐片吐文本。TTFT 就是
"从发起请求到**第一片非空文本**到达"的时间。思路：

1. 开始计时；
2. `for piece in stream_fn():` 逐片收；
3. **第一次收到非空片段时**记下时刻差——`if ttft is None and piece:`
   （判 `piece` 非空很关键：有些实现会先吐空片占位）；
4. 顺手把所有片段拼成完整文本，和 TTFT 一起返回
   （调用方常常想顺便看看生成了什么）。

参考答案的全部逻辑（先自己写！）：

```python
    start = time.perf_counter()
    ttft = None
    pieces = []
    for piece in stream_fn():
        if ttft is None and piece:
            ttft = time.perf_counter() - start
        pieces.append(piece)
    return (ttft if ttft is not None else time.perf_counter() - start), "".join(pieces)
```

最后那个三元表达式是兜底：万一流空了（一片都没吐），TTFT 就记为总耗时，
不让它留在 `None`。

> ⚠️ **易踩坑：** 两个高频错误——
> ① 把 `time.perf_counter() - start` 写成了只调 `time.perf_counter()`
> （那是"从某个远古起点到现在的秒数"，不是间隔）；
> ② 忘了判 `piece` 非空，空片段把 TTFT 记成了 0。

## 4.5 用法里的学问：先预热，再计时

看一眼第 5 章要用的实测脚本 `scripts/bench_minivllm.py` 的关键几行：

```python
    run(PROMPTS[:1])  # 预热
    ttft, _ = measure_ttft(lambda: llm.stream(PROMPTS[0], params))
    results = {
        "minivllm_b1_tps": round(run(PROMPTS[:1]), 1),
        "minivllm_b8_tps": round(run(PROMPTS), 1),
        "minivllm_ttft_ms": round(ttft * 1000),
    }
```

注意第一行：**正式测量之前，先白跑一次**。为什么？因为第一次运行有一堆
"一次性开销"：模型权重进显存、MPS 编译计算图、各种缓存初始化——这些
属于"开机"，不属于"跑速"。不测预热，你量的就是"开机 + 跑"，数字会难看且不稳定。

另外两处细节：

- `lambda: llm.stream(PROMPTS[0], params)`——`measure_ttft` 要的合同是
  "无参 callable 返回迭代器"，而 `llm.stream` 要带参数，所以用 `lambda`
  包一层。**计时从 lambda 被调用的那一刻起算**，这正是我们想要的"从发起开始"；
- `ttft * 1000` 换成毫秒——人对"62 毫秒"有感觉，对"0.062 秒"没感觉。

文件里还有一个已给好的 `format_report(result)`，把成绩单排成好读的对齐文本，
第 5 章跑完实测会用到它。

## 4.6 用假函数跑绿测试

`tests/test_w8.py` 的巧思：**全程不碰真模型**。它塞进来的都是假函数，
比如测 TTFT 的这条：

```python
def test_measure_ttft_and_text():
    def fake_stream():
        time.sleep(0.01)
        yield "你"
        yield "好"
        yield "吗"

    ttft, text = metrics.measure_ttft(fake_stream)
    assert 0.01 <= ttft < 1.0
    assert text == "你好吗"
```

`fake_stream` 先睡 10 毫秒再吐第一片——所以 TTFT 必须落在
$[0.01, 1.0)$ 区间：下界检查你确实等到了第一片才停表
（没等就是 0），上界检查你没把睡第二觉的工夫也算进去。
`text == "你好吗"` 检查你拼了完整文本。你的实现对不对，这条测试一眼看穿。

测吞吐的 `test_benchmark_throughput_aggregates` 同理：假 `generate` 睡 10 毫秒、
每个 prompt 返回 10 个假 token，检查你数对了 `3` 个 prompt、`30` 个 token、
总耗时不少于 0.01 秒。

> 📌 **划重点：** TTFT 管"第一下的体验"（单请求流式测），吞吐管"持续的能力"
> （整批测）。两支体温计都不认识引擎——只认 callable。测量铁律：先预热，再计时。

---

## 动手练习

1. 在 `minivllm/bench/metrics.py` 里实现 `benchmark_throughput`
   （照 4.3 的三步，先别看答案）；
2. 实现 `measure_ttft`（照 4.4，注意"非空片段"和兜底）；
3. 跑：

```bash
.venv/bin/pytest tests/test_w8.py
```

   4 条测试应该全绿。再跑 `IMPL=reference .venv/bin/pytest tests/test_w8.py`
   对一下参考答案的行为。
4. （选做）写三行代码，用你实现的 `benchmark_throughput` 给
   `tests/test_w8.py` 里的 `fake_generate` 测一次，亲手打印一份
   `format_report`——你会看到和测试断言一样的数字。

## 参考答案

`reference/bench/metrics.py` 是标准答案，两个函数加起来不到 20 行。
**卡住 20 分钟再看**，重点对照：TTFT 的"非空"判断写了没有、
兜底的三元表达式处理了没有。

---

👉 下一章：[第 5 章：结业实测——我们追到了几成功力](./05-结业实测-我们追到了几成功力.md)
