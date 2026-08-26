# 第 2 章：带读 vLLM 调度器源码

> 本章你将：
> 1. 逐段读完真实 `vllm/v1/core/sched/scheduler.py` 里 `schedule()` 的开篇；
> 2. 在真实 `RequestStatus` 里认出我们的三态状态机（和它多出来的"成人烦恼"）；
> 3. 看懂真实块池 `get_new_blocks()` 比我们的 `allocate()` 多做的两件事；
> 4. 拿到一张"我们的 ↔ 真实的"逐条对照表，以后自己读源码不迷路。

所有引用都来自你本地 clone 的仓库 `$VLLM_SRC`，行号是写稿时的真实行号
（源码会演进，对不上几行很正常，用 `grep` 按函数名找）。

---

## 2.1 读之前的约定：只盯三样东西

Week 5 你写的 `Scheduler.schedule()` 大约 40 行；真实版 2900 行的文件里，
`schedule()` 一个方法就有几百行。别怕——**第一遍只盯三样东西**：

1. **队列**：waiting / running 在哪，谁从哪边挪到哪边；
2. **预算**：这一步最多算多少个 token（token budget）；
3. **分块**：什么时候给请求发新块（`allocate_slots` / `get_new_blocks`）。

其余的一切——encoder 预算、LoRA 约束、KV connector、流水线并行——
都是这三件主干的"挂件"。第一遍读到不认识的分支，直接跳过。

## 2.2 第一站：`RequestStatus`——我们的三态长大了

打开 `vllm/v1/request.py`，找到第 351 行：

```python
class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()
```

（`$VLLM_SRC/vllm/v1/request.py`，第 351~367 行）

我们的是 3 个状态，真实的是 13 个。但拆开来全是熟人：

| 真实状态 | 对应我们的 | 多出来的部分在管什么 |
|---|---|---|
| `WAITING` | `WAITING` | 一模一样 |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 等三个 | （没有） | 等语法编译、等远端 KV 送来、等流式输入——分布式世界的等待 |
| `RUNNING` | `RUNNING` | 一模一样 |
| `PREEMPTED` | （没有） | **被抢占**：内存不够时被踢回等待席（下面 2.4 会看到案发现场） |
| `FINISHED_STOPPED` | `FINISHED`（`finish_reason="stop"`） | 我们的 `finish_reason` 字段，真实版直接编码进状态名 |
| `FINISHED_LENGTH_CAPPED` | `FINISHED`（`finish_reason="length"`） | 同上 |
| `FINISHED_ABORTED / IGNORED / ERROR / REPETITION` | （没有） | 被用户取消、提示词超长、出错、重复检测 |

再看它自带的一个静态方法（第 372~374 行）：

```python
    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED
```

**一行代码白捡一个设计**：状态用 `IntEnum` 按顺序编号，"是否结束"就变成一次
大小比较——`PREEMPTED` 之后的全部算结束。我们的 `Request.is_finished()`
是自己比状态，思想相同，真实版用编号顺序把判断写得更省。

## 2.3 `schedule()` 开篇：一段值得背诵的注释

打开 `vllm/v1/core/sched/scheduler.py` 第 440 行，`schedule()` 的开门见山不是代码，
是作者 Woosuk（vLLM 核心作者）写的一段注释（第 442~451 行）：

```python
    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
```

这段话翻译成人话：

> **调度器里根本没有"prefill 阶段"和"decode 阶段"之分。** 每个请求只有
> 两个数字：`num_computed_tokens`（已经算到第几个 token）和
> `num_tokens_with_spec`（现在总共有多少个 token 该有）。
> 每一步，调度器给请求们分派 token，让前者追上后者。

我们的 `Scheduler` 把 prefill 和 decode 显式拆成两批（`SchedulerOutput` 的
`.prefill` 和 `.decode`），是为了让第一次读的人看得懂。真实版把它俩统一成
"还差几个 token 没算"——这个视角更一般，**chunked prefill、前缀缓存、
投机采样全都是这一个公式的特例**。下一章讲 chunked prefill 时你会立刻用到它。

## 2.4 先排 RUNNING：预算、准入、抢占

紧接着（第 453~460 行）是每步的"账本"初始化：

```python
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
```

最后那行是**本周最重要的新概念**：`token_budget`，本步允许计算的总 token 数
（来自配置 `max_num_scheduled_tokens`，第 111~115 行）。我们 Week 5 的准入条件是
"块池装得下吗"（保守准入）；真实版在此之上还加了一条"本步的 token 预算花完了吗"。

然后是主循环——**先伺候已经在跑的 running 队列**（第 484~486 行）：

```python
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
```

和我们 `schedule()` 的骨架一模一样：running 优先，waiting 候补。循环体里，
对每个请求算"这一步该补几个 token"（第 517~524 行）：

```python
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
```

三个数字一目了然：**该补的 = 总共该有的 − 已经算了的**（正是 2.3 那段注释的
公式落地），然后被两个"上限"砍一刀：长 prefill 阈值和本步剩余预算。
砍一刀的结果就是**长提示词可以分好几步算完**——这就是 chunked prefill，
第 3 章细讲。

接下来是发块（第 576~587 行）：

```python
            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break
```

`allocate_slots` 返回 `None` 就是"块不够了"。我们的版本此时会怎么办？
Week 5 的答案是：**新请求不许进**（保守准入，在 waiting 里继续等）。
真实版激进得多——它回头从 running 队列的尾巴上**踢一个请求出去**，
把它的块收回来（第 590~616 行，节选）：

```python
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        ...
                    else:
                        preempted_req = self.running.pop()
```

这就是**抢占（preemption）**：块不够时，把优先级最低（或最晚来的）的请求
踢回等待席，释放它的 KV 块给更前面的请求用，等内存宽裕了再把它捞回来重算。
代价是被踢的请求要重来一段 prefill，好处是永远不会因为"一时没块"而死锁。

> 💡 **你可能会问：为什么我们不做抢占？**
>
> 因为保守准入在"请求规模可预估"的小场景里更简单、更好懂，而且绝不会
> 白算。抢占是真实流量下的必需品：请求长短不可预知，与其把新请求关在门外，
> 不如请跑得动的先跑。两种策略没有对错，是"教学版"和"生产版"的分工。

循环收尾（第 632~638 行）记账：该请求排上了，预算扣掉它的 token 数：

```python
            # Schedule the request.
            scheduled_running_reqs.append(request)
            ...
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
```

## 2.5 再排 WAITING：容量闸门与前缀缓存的钩子

running 全部安排完，才轮到 waiting（第 684~698 行，节选）：

```python
        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            ...
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                ...
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break
```

两道闸门，你都认识：

- `token_budget > 0`——本步预算还有剩吗（我们没有这条）；
- `num_running >= self.max_num_running_reqs`——**这就是我们的 `max_num_seqs`！**
  配置项 `max_num_seqs` 在第 110 行被存成 `max_num_running_reqs`，
  一周前你写的 `Scheduler(block_pool, max_num_seqs=8)` 在这里有同款旋钮。

waiting 循环里还藏着一个第 3 章的主角（第 746~767 行，节选）：

```python
                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    ...
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            ...
                        ) = self.kv_cache_manager.get_computed_blocks(request)
```

新请求准入前，先问一句"你的开头是不是别人算过了？"——`get_computed_blocks`
就是**前缀缓存**的查询入口，命中了就不用重算提示词前面那截。先记住这个位置，
第 3 章专门讲它。

## 2.6 第三站：`get_new_blocks`——块池的成人礼

最后一站，`vllm/v1/core/block_pool.py` 第 647 行。对照你 Week 4 写的
`BlockPool.allocate()`（从 `_free` 队列 `popleft` 一个块号），真实版一次发一串：

```python
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                ...
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                ...
        return ret
```

（`$VLLM_SRC/vllm/v1/core/block_pool.py`，第 647~677 行）

骨架和我们的 `allocate()` 分毫不差：**空闲队列 `free_block_queue` 弹出来
（`popleft_n` 就是批量版 `popleft`），池空就报错**。多出来的两样东西：

1. **`ref_cnt`（引用计数）**：块不再"非空闲即在用"，而是数"有几个人在用我"。
   为什么？因为前缀缓存会让**两个请求共享同一个块**（相同的开头，KV 直接复用），
   计数到 0 才真空闲。我们的版本一个块同一时刻只属于一个请求，所以不需要计数；
2. **`_maybe_evict_cached_block`**：发出去的块可能还挂着"缓存条目"（它装的
   内容曾被哈希登记过），发新主人之前先把旧登记抹掉——**逐出（eviction）**。
   这也是为前缀缓存服务的，第 3 章细讲。

`material/vllm-源码对照.md` 里的那句话现在可以落地了：真实块池比我们多的
两件大事就是**前缀缓存**和**逐出**，我们的 BlockPool 是去掉它们后的骨架——
而发块/收块的骨架，你早就写过了。

## 2.7 一张总对照表

把这章读到的逐条对上：

| 我们 Week 5/4 写的 | 真实 vLLM v1 | 差别 |
|---|---|---|
| `RequestStatus` 三态 | 13 态（`request.py:351`） | 多了等待细分、抢占、多种结束原因 |
| `finish_reason` 字段 | 编码进状态名 + `_FINISHED_REASON_MAP` | 一处信息，两种放法 |
| `schedule()` 分 prefill/decode 两批 | 统一成"补 token"（`scheduler.py:442` 注释） | 真实版视角更一般 |
| 保守准入：块池装得下才放行 | token budget + 容量闸门 + **抢占**（`scheduler.py:590`） | 真实版敢踢人 |
| `max_num_seqs=8` | `max_num_running_reqs`（`scheduler.py:110`） | 同款旋钮 |
| `BlockPool.allocate()` 发一块 | `get_new_blocks()` 发一串 + 引用计数 + 逐出（`block_pool.py:647`） | 为前缀缓存服务 |

> ⚠️ **易踩坑：** 行号会漂移。本章给的行号是写稿时本机仓库的真实行号，
> vLLM 迭代很快，你打开时对不上几行甚至几十行都正常——**按函数名 grep**
> （`grep -n "def schedule" ...`）才是永恒的定位方式。

> 📌 **划重点：** 真实 `schedule()` = 我们的骨架（running 优先、waiting 候补、
> 容量闸门）+ 三件新武器（token budget、抢占、前缀缓存钩子）。骨架你写过了，
> 新武器下一章讲两个最重要的。

---

## 动手练习

1. 自己把 2.3 节那段 `NOTE(woosuk)` 注释在真实文件里找出来读一遍原文：

```bash
grep -n -A 10 "NOTE(woosuk) on the scheduling algorithm" \
  $VLLM_SRC/vllm/v1/core/sched/scheduler.py
```

2. 在真实 `scheduler.py` 里找到 `token_budget -= num_new_tokens` 这一行
   （提示：在 `schedule()` 的 running 循环收尾处），用自己的话说出：
   如果 `token_budget` 设得特别小（比如 1），系统会发生什么？
3. （思考题）2.2 节里 `is_finished` 用 `status > RequestStatus.PREEMPTED`
   一次比较搞定判断。如果我们的 `RequestStatus` 也想偷这个设计，
   枚举成员的排列顺序应该怎么定？

## 参考答案

本章的"答案"都在真实源码里，路径与行号已随文给出；阅读顺序和骨架说明见
`material/vllm-源码对照.md`。第 2 题提示：想想 prefill 一个 64 token 的
提示词要分多少步。第 3 题提示：把"结束类"状态全部排在某个界限成员之后。

---

👉 下一章：[第 3 章：高级特性导览——prefix caching 与 chunked prefill](./03-高级特性导览-prefixcaching与chunkedprefill.md)
