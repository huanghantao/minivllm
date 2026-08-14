# 第 3 章：Scheduler——每步点名

> 本章你将：
> 1. 实现 `_can_admit`：批次有空位 **且** 块池装得下，才放请求进门；
> 2. 实现 `schedule()`：每步点名，产出两张名单——谁 prefill、谁 decode；
> 3. 实现 `update_after_step()`：登记新词、判断下车、还块销账；
> 4. 说清"保守准入"用 `_reserved_blocks` 记账的取舍——为什么我们的请求
>    永远不用担心算到一半没内存；
> 5. 跑绿 `pytest tests/test_w5.py -k scheduler`。

---

## 3.1 调度员的一天

把调度器想成餐厅叫号员，它的工作日就是无限重复两件事：

1. **点名**（`schedule()`）：看一眼等位区，有空桌、食材够，就叫号让人入座；
   然后写出本步的两张名单——**新入座的去点菜（prefill），已经在吃的继续吃
   （decode）**；
2. **收盘**（`update_after_step()`）：本步吃完一轮后，检查每桌：吃完的结账
   收桌，桌子还回"空桌池"，等位的人下一步才能进来。

两张名单装在一个小盒子里，`SchedulerOutput`（已给出）：

```python
class SchedulerOutput:
    """一次点名的结果：这一步该谁 prefill、该谁 decode。"""

    def __init__(self):
        self.prefill = []  # 新准入的请求（本步跑 prefill + 采第一个词）
        self.decode = []   # 已经在跑的请求（本步各解一个 token）

    def is_empty(self):
        return not self.prefill and not self.decode
```

为什么是两张名单而不是一张？因为两者的"吃法"完全不同（Week 3 的老知识）：
prefill 一次吃掉整段提示词；decode 每步只吃一个词。模型为此专门长了两个接口
（Week 4 的 `prefill` / `decode_batch`），名单自然也要分开。

## 3.2 家底：三个容器 + 一本账

`Scheduler.__init__`（已给出）：

```python
def __init__(self, block_pool, max_num_seqs=8):
    self.block_pool = block_pool
    self.max_num_seqs = max_num_seqs
    self.waiting = deque()   # Request 队列，先到先服务
    self.running = []        # 正在生成的 Request
    self.finished = []       # 已结束的 Request
    self._reserved_blocks = 0  # 已准入请求"预订"的块数
```

三个容器上一章见过。新面孔是 `_reserved_blocks`——**一本预订账**。

这里有个微妙的错位，是整个调度器设计的核心，务必看懂：

- 请求一被准入，我们就要保证它**全程**（提示词 + 全部待生成词）都有块可用；
- 但物理块不是一次性领走的——Week 4 的 `BlockTable.append_slot()` 是
  **按需**分配，生成到第几个词才领第几个块。

"承诺"和"实领"之间隔着整个生成过程。如果不记账，就会出事故：A 准入时池子
很空，B 也准入，C 也准入……生成到一半，三个人同时要新块，池子空了，谁死？
所以准入那一刻就把"全程要用的块数"记在账上，后来的请求看到的可用块数要
**先减去别人预订的**。物理块按需领，账先记上——这就是 `_reserved_blocks`。

## 3.3 保守准入：_can_admit

先算"一个请求全程需要几块"（`_blocks_needed`，已给出，读懂即可）：

```python
@staticmethod
def _blocks_needed(request, block_size):
    total = request.num_prompt_tokens + request.sampling_params.max_new_tokens
    return -(-total // block_size)  # 向上取整
```

`-(-total // block_size)` 是"向上取整除法"的经典写法：`//` 是向下取整的
整除，套两次负号就变成向上取整。比如 9 个 token、块大小 4：$\lceil 9/4 \rceil = 3$
块（$9 = 4 + 4 + 1$，最后 1 个也要独占一块）。

你的第一个任务，`_can_admit`——同时满足两个条件才放行：

```python
def _can_admit(self, request):
    """保守准入：同时满足"批次有空位"和"块池装得下全程"才放行。"""
    if len(self.running) >= self.max_num_seqs:
        return False                                  # 条件一：座位满了
    needed = self._blocks_needed(request, self.block_pool.block_size)
    available = self.block_pool.num_free_blocks() - self._reserved_blocks
    return needed <= available                        # 条件二：块够装下全程
```

注意 `available` 的算法：**池子里现在空闲的块，减去别人已预订的**。预订的块
虽然还躺在池子里（还没被真正领走），但账上已经不属于后来人了。

> 💡 **你可能会问：按"最大生成长度"记账是不是太悲观了？很多请求根本生成不满，
> 预订的块不就浪费了吗？**
>
> 对，这就是"保守"二字的代价：块被多订了，能同时服务的请求数变少。但换来一个
> 巨大的简化——**运行中途绝不会缺块**，请求一旦上车就保证坐到终点，所以我们
> 不需要真实 vLLM 那套复杂的"抢占"机制（块不够时把请求踢回队列）。这是教学
> 上非常划算的取舍，第 6 章会对比。

## 3.4 schedule()：点名一次

你的第二个任务。逻辑就一句话：**队首能进就进，进到进不动为止**。

```python
def schedule(self):
    """点名一次，返回 SchedulerOutput。"""
    out = SchedulerOutput()
    while self.waiting and self._can_admit(self.waiting[0]):
        req = self.waiting.popleft()
        req.status = RequestStatus.RUNNING
        req.block_table = BlockTable(self.block_pool)
        req.reserved_blocks = self._blocks_needed(req, self.block_pool.block_size)
        self._reserved_blocks += req.reserved_blocks
        self.running.append(req)
        out.prefill.append(req)
    out.decode = [r for r in self.running if r not in out.prefill]
    return out
```

逐步看：

1. **只看队首**（`self.waiting[0]`）：先到先服务，不许插队。队首进不来
   （座位满或块不够），后面的人再小也不放——这是 FIFO 的规矩；
2. **准入六件套**：出队 → 改状态 RUNNING → 发块表（此刻才 `BlockTable` 到手，
   上周学的"页表"正式上岗）→ 算预订块数 → 记账 → 进 `running` 和
   `out.prefill`；
3. **decode 名单**：所有在跑的、**减去**刚准入的。刚准入的本步只 prefill，
   prefill 时就已经采出第一个词了，它的第一次 decode 在下一步。

> ⚠️ **易踩坑：** `out.decode` 千万别写成 `[r for r in self.running]`——
> 那样新准入的请求会既 prefill 又 decode，一步被算两次，KV 也写重了。
> 测试 `test_scheduler_fifo_admission` 第二步专门检查这件事：准入那一步
> `out.decode == []`。

## 3.5 update_after_step()：收盘

本步模型算完、每个跑过的请求都采出了一个新词（装在一个字典
`{request_id: token_id}` 里传进来），调度器要收盘。你的第三个任务：

```python
def update_after_step(self, sampled_tokens, eos_token_ids=()):
    """登记本步采出的 token，把做完的请求请下车。返回本步刚结束的请求列表。"""
    newly_finished = []
    for req in list(self.running):            # 注意：先复制一份再遍历
        if req.request_id not in sampled_tokens:
            continue
        token = sampled_tokens[req.request_id]
        req.output_token_ids.append(token)    # ① 登记新词

        hit_length = req.num_output_tokens >= req.sampling_params.max_new_tokens
        hit_stop = token in eos_token_ids or token in req.sampling_params.stop_token_ids
        if hit_length or hit_stop:            # ② 判断下车
            req.status = RequestStatus.FINISHED
            req.finish_reason = "length" if hit_length else "stop"
            req.block_table.free()            # ③ 下车三件套：还块
            self._reserved_blocks -= req.reserved_blocks   # 销账
            self.running.remove(req)                       # 挪去归档
            self.finished.append(req)
            newly_finished.append(req)
    return newly_finished
```

三个关键点：

1. **`for req in list(self.running)`**：循环体里会 `self.running.remove(req)`——
   **边遍历列表边删元素是 Python 经典翻车现场**（会跳过元素）。先 `list(...)`
   复制一份快照再遍历，就安全了；
2. **不在字典里的请求跳过**：`sampled_tokens` 只包含本步跑过的请求。某个
   running 请求本步没被点到名（理论上我们的排法不会发生，但防御一下没坏处）；
3. **下车三件套一个都不能少**：`block_table.free()` 把物理块还回池子、
   `_reserved_blocks` 销账、`running` 挪到 `finished`。少了任何一个，
   后面的请求就会莫名其妙地永远进不来——块在账上被"幽灵"占着。

## 3.6 拿测试当剧本走一遍

四条测试，每条钉死一种行为：

**`test_scheduler_fifo_admission`**：3 个请求、`max_num_seqs=2`。
第一步只准入 0、1 号（座位满了），`prefill=[0,1]`、`decode=[]`、waiting 剩 1 个；
第二步 0、1 号转入 decode 名单，`prefill=[]`、`decode=[0,1]`。

**`test_scheduler_conservative_admission`**：池子只有 2 块 × 4 = 8 格。
0 号要 $3 + 5 = 8$ 格，$\lceil 8/4 \rceil = 2$ 块，**正好**够，准入；
1 号要 $3 + 6 = 9$ 格，$\lceil 9/4 \rceil = 3$ 块，池子只剩 0 块空闲
（2 块被 0 号预订），进不来。**一格之差，一进一拒**——这条测试就是冲着
你的向上取整和预订账来的。

**`test_scheduler_finish_frees_blocks`**：0 号准入后真的占了 1 个物理块，
生成满 2 词后下车——断言 `pool.num_free_blocks()` **恢复**了。这条专抓
"忘了还块/忘了销账"。

**`test_scheduler_stop_token`**：`max_new_tokens=10` 还很宽裕，但本步采出的
词 `7` 在 `eos_token_ids` 里——以 `"stop"` 提前下车。

> 📌 **对标 vLLM：** 我们的 `schedule()` 对应 `vllm/v1/core/sched/scheduler.py`
> 的同名方法：一样是"从 waiting 队首尝试准入、running 里的继续跑"。真实版的
> 准入判断更复杂——按 **token 预算**（本步最多算多少个 token）而不是按请求数
> 卡点，还可能在 running 没资源时把它**抢占**回 waiting。第 6 章细说。

> 📌 **划重点：** 调度器每步做两件事：`schedule()` 点名（队首 FIFO 准入，
> 新准入的 prefill、老的 decode），`update_after_step()` 收盘（登记新词，
> 够长或撞结束符就下车：还块、销账、归档）。保守准入 = 准入时把全程的块
> 记在 `_reserved_blocks` 账上，换来"中途绝不缺块"。

---

## 动手练习

1. 填出 `minivllm/scheduler/scheduler.py` 里的三个方法：`_can_admit`、
   `schedule`、`update_after_step`（照 3.3–3.5 的思路，先自己写）；
2. 跑：

```bash
.venv/bin/pytest tests/test_w5.py -k scheduler
```

3. （思考题）把 `test_scheduler_conservative_admission` 里的池子改成
   `num_blocks=3`：0 号、1 号能同时准入吗？心算出答案后改一改测试数据、
   在 REPL 里验证（别改测试文件本身）。

## 参考答案

`reference/scheduler/scheduler.py`，三个方法加起来不到 50 行。
**卡住 20 分钟再看**，重点对照：`_can_admit` 里有没有减 `_reserved_blocks`；
`schedule` 里 decode 名单有没有排除刚准入的；`update_after_step` 遍历时
有没有 `list(...)` 复制。

---

👉 下一章：[第 4 章：组装流水线——continuous_batch_generate](./04-组装流水线-continuous_batch_generate.md)
