# 第 2 章：Request 与两个队列——waiting 与 running

> 本章你将：
> 1. 认识引擎里流转的"工单" `Request`——它身上记了哪些账；
> 2. 画出请求的三态状态机：WAITING → RUNNING → FINISHED；
> 3. 搞懂两种 `finish_reason`："length"（够长了）和 "stop"（撞见结束符）；
> 4. 跑绿 `pytest tests/test_w5.py -k request`。

---

## 2.1 类比：一张跟到底的等位单

去火爆的餐厅吃饭，你会拿到一张等位单。这张单子从你进门跟到你出门：

- **等位中**：单子在叫号屏上排队；
- **上桌中**：叫到你的号，入座开吃；
- **已离店**：吃完结账，单子作废归档。

调度器就是餐厅的叫号员。它每"点名"一次，就看一眼所有单子：谁的号到了、
有空桌就入座；谁吃完了，结账收桌，桌子让给排队的人。

`Request` 就是这张单子。一个请求从进门到出门经历三种状态，代码里用枚举写得
明明白白（`minivllm/scheduler/request.py`）：

```python
class RequestStatus(Enum):
    WAITING = auto()    # 排队中
    RUNNING = auto()    # 在算
    FINISHED = auto()   # 做完了
```

状态机长这样（箭头上的字是"谁推动了这次跳转"）：

```
WAITING ──调度器准入──▶ RUNNING ──生成够长 / 撞见结束符──▶ FINISHED
```

注意箭头只有**前进**没有后退——我们的迷你调度器里，请求一旦上桌就不会被
赶回等位区（真实 vLLM 会，那叫"抢占"，第 6 章讲）。

## 2.2 一张工单上记了什么

`Request` 本周**已经完整给出**（你不用填），但每个字段都要读懂，因为下一章
写调度器时全靠这些账：

```python
class Request:
    def __init__(self, request_id, prompt_token_ids, sampling_params=None):
        assert len(prompt_token_ids) > 0, "提示词不能为空"
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.status = RequestStatus.WAITING   # 进门先排队
        self.output_token_ids = []
        self.block_table = None               # 准入后才发"页表"
        self.reserved_blocks = 0              # 准入时向块池预订的块数
        self.finish_reason = None
```

一张表看懂每个字段：

| 字段 | 是什么 | 什么时候有值 |
|---|---|---|
| `request_id` | 工单号 | 进门时 |
| `prompt_token_ids` | 提示词（用户给的） | 进门时 |
| `sampling_params` | 采样参数表（最多生成多少词、结束符等） | 进门时 |
| `status` | 三态之一 | 全程变化 |
| `output_token_ids` | 已经生成出来的词 | 逐步变长 |
| `block_table` | 它的 KV cache"页表"（Week 4 的 `BlockTable`） | **被调度器准入后**才发放 |
| `reserved_blocks` | 准入时向块池"预订"的块数 | 准入时记账，下车时销账 |
| `finish_reason` | 结束原因 | 下车那一刻 |

`sampling_params` 是 Week 6 才正式登场的东西，本周你只需要知道它的两个字段：
`max_new_tokens`（这个请求最多生成几个词，默认 16）和 `stop_token_ids`
（额外的结束符集合）。它的 `temperature` 默认 0.0，意思是**贪心**——永远挑
分数最高的词，所以本周的生成是完全确定、可复现的。

## 2.3 几个会算账的属性

`Request` 上有四个小方法/属性，调度器和流水线每一步都在用：

```python
@property
def num_prompt_tokens(self):
    return len(self.prompt_token_ids)

@property
def num_output_tokens(self):
    return len(self.output_token_ids)

@property
def num_tokens(self):
    """它一共占多少个 KV 槽位（提示词 + 已生成）。"""
    return self.num_prompt_tokens + self.num_output_tokens

def last_token_id(self):
    """下一步要喂给模型的那个 token。"""
    if self.output_token_ids:
        return self.output_token_ids[-1]
    return self.prompt_token_ids[-1]

def is_finished(self):
    return self.status is RequestStatus.FINISHED
```

逐个说：

- `num_tokens`：这个请求**此刻**占了多少个 KV 槽位。准入时要靠它估算
  "块池装不装得下"（下一章的准入判断）；
- `last_token_id()`：**下一步喂给模型的那个词**。已经生成过词，就是最新生成的
  那个；一个词都还没生成，就是提示词的最后一个词。Week 3 你就知道：
  decode 每步只喂最新一个词；
- `is_finished()`：看状态是不是 FINISHED。

`tests/test_w5.py` 里的 `test_request_basics` 把这本账钉死了，读一遍：

```python
req = make_request(0, [10, 20, 30])     # 提示词 3 个词，最多生成 4 个
assert req.num_prompt_tokens == 3
assert req.num_tokens == 3
assert req.last_token_id() == 30        # 还没生成过：喂提示词最后一个词
assert not req.is_finished()
req.output_token_ids.append(99)         # 假装生成了一个词 99
assert req.last_token_id() == 99        # 现在喂最新生成的词
assert req.num_tokens == 4              # 3 + 1 = 4 个槽位
```

## 2.4 finish_reason：两种"下车原因"

请求什么时候算"做完"？两种情况，对应 `finish_reason` 的两个取值：

| finish_reason | 触发条件 | 生活类比 |
|---|---|---|
| `"length"` | 已生成词数达到 `max_new_tokens` | 自助餐时间到，掐表请走 |
| `"stop"` | 刚生成的词撞进结束符集合 | 客人自己说"我吃饱了" |

结束符集合有两个来源，任一命中都算：调用方传的 `eos_token_ids`（全局结束符，
比如分词器的句号 token），以及这个请求自己 `sampling_params.stop_token_ids`
里登记的。具体判断逻辑在下一章的 `update_after_step` 里，这里先记住结论：
**不管是哪种下车，原因都会记在 `finish_reason` 上**，调用方可以据此区分
"话说完了"还是"被截断了"。

> 💡 **你可能会问：如果同一步既够长了又撞见结束符，算哪种？**
>
> 我们的实现里 `length` 优先（代码先判断 `hit_length`）。两种都对，只要规则
> 固定、可预期就行——测试 `test_scheduler_stop_token` 和
> `test_scheduler_finish_frees_blocks` 各验证了一种纯粹的下车方式。

## 2.5 两个队列（外加一个归档盒）

调度器手里有三个容器（下一章逐个用）：

| 容器 | 类型 | 装什么 |
|---|---|---|
| `waiting` | `deque`（双端队列） | 排队的请求，**先到先服务** |
| `running` | `list` | 正在生成的请求（一批座位的客人） |
| `finished` | `list` | 已结束的请求（归档，方便调用方取结果） |

`deque` 是 Python 标准库 `collections` 里的双端队列，我们用它的两个操作：
`append`（新请求排到队尾）和 `popleft`（从队首取出最早来的那个）——这就是
"先进先出"（FIFO），排队叫号的标准动作。

> 📌 **对标 vLLM：** 我们的 `Request` 对应真实 vLLM 的 `vllm/v1/request.py`，
> `RequestStatus` 的取值和它一一对应（真实版还多一个 `PREEMPTED` 被抢占态）。
> waiting/running 两个队列的名字也和 `vllm/v1/core/sched/scheduler.py` 里
> 一模一样——Week 8 读源码时你会有一种"回家"的感觉。

> 📌 **划重点：** `Request` 是跟着请求走全程的工单：记提示词、记已生成的词、
> 记状态（WAITING/RUNNING/FINISHED）、记块表、记下车原因。调度器靠
> waiting/running 两个队列管理这些工单，先到先服务。

---

## 动手练习

1. 通读 `minivllm/scheduler/request.py`（不到 70 行，全部已给出），
   对照 2.2 的字段表，确认每个字段你都找到了；
2. 跑：

```bash
.venv/bin/pytest tests/test_w5.py -k request
```

3. （REPL 里玩一玩）用 `.venv/bin/python` 启动解释器，构造一个提示词
   `[5, 6, 7]`、`max_new_tokens=2` 的 `Request`，手动 `append` 两个词进
   `output_token_ids`，每步打印 `req`（它的 `__repr__` 会把状态、提示词长度、
   已生成长度都打出来），观察账目的变化。

## 参考答案

`Request` 本周不需要你填代码，所以没有"答案"要偷看。想确认理解是否到位，
对照 `reference/scheduler/request.py`——它和 `minivllm/` 里的版本一字不差。
**卡住 20 分钟再看**的原则本周适用于下一章的 `Scheduler`。

---

👉 下一章：[第 3 章：Scheduler——每步点名](./03-Scheduler-每步点名.md)
