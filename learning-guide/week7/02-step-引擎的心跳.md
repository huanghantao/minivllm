# 第 2 章：step——引擎的心跳

> 本章你将：
> 1. 看懂 `step()` 的五拍：点名 → prefill 新人 → 批量 decode → 采样 → 登记下车；
> 2. 发现一个大秘密：`step()` 的循环体你 Week 5 就写过，本周只做两处"换血"；
> 3. 亲手实现 `LLMEngine.step()`；
> 4. 搞懂两个容易绕进去的细节：为什么 prefill 的请求当步就出第一个词，
>    以及 `positions` 为什么要减 1。

---

## 2.1 心跳是什么

上一章说过，引擎和函数的区别是"收件与开工分离"。那开工的节奏由谁定？
答案是 `step()`：**调用一次，引擎往前走一步**——

> 点名一次 → 本步新来的请求各做一次 prefill → 所有在跑的请求批量 decode 一步 →
> 给每个跑过的请求采一个新词 → 把新词登记到请求身上，做完的请下车。

调用方拿着 `step()` 就像拿着发动机的手柄：`while` 循环里反复摇，引擎就一直转。
摇得快慢、什么时候停，全由调用方决定（第 3 章的 `generate` 和 `stream` 就是两种
不同的"摇法"）。

这个"一步"的粒度是精心选过的：**一步里既有 prefill 又有 decode**。这正是 Week 5
continuous batching 的精髓——新请求不用等老请求全做完才上车，老请求也不用陪
新请求重跑提示词，大家在同一次心跳里各取所需。

## 2.2 和 Week 5 对一对：其实只有两处换血

把 Week 5 `continuous_batch_generate` 的循环体和本周的 `step()` 摆在一起，
你会发现**骨架一模一样**，只有两处变化：

| | Week 5 的循环体 | Week 7 的 `step()` |
|---|---|---|
| 采样方式 | `int(torch.argmax(logits).item())`——写死贪心 | `self.sampler.sample_one(logits, rid, req.sampling_params)`——每个请求一套参数 |
| 结束符 | 函数参数 `eos_token_ids` | `self.eos_token_ids`（构造引擎时定好） |
| 循环条件 | 函数里包着 `while` | 没有 `while`！循环交给了调用方 |

换句话说：**Week 6 写的采样器正式上岗**，替换掉 Week 5 的临时贪心；
而 `while` 从函数肚子里挪到了外面——这一挪，函数就变成了引擎。

## 2.3 step() 逐拍拆解

这是你要实现的完整函数（`minivllm/engine/llm_engine.py`）。先通读，再逐拍讲：

```python
@torch.no_grad()
def step(self):
    """走一步：点名 → prefill 新请求 → 批量 decode → 登记结果。

    返回本步刚完成的 request_id 列表。
    """
    out = self.scheduler.schedule()
    sampled = {}

    for req in out.prefill:
        slots = [
            req.block_table.append_slot()
            for _ in range(req.num_prompt_tokens)
        ]
        ids = torch.tensor(
            req.prompt_token_ids, dtype=torch.long, device=self.device
        )
        logits = self.model.prefill(ids, self.kv_cache, slots, req.block_table)
        token = self.sampler.sample_one(
            logits, req.request_id, req.sampling_params
        )
        sampled[req.request_id] = token

    if out.decode:
        slots = [req.block_table.append_slot() for req in out.decode]
        token_ids = torch.tensor(
            [req.last_token_id() for req in out.decode],
            dtype=torch.long, device=self.device,
        )
        positions = torch.tensor(
            [req.block_table.num_tokens - 1 for req in out.decode],
            dtype=torch.long, device=self.device,
        )
        logits = self.model.decode_batch(
            token_ids, positions, self.kv_cache, slots,
            [req.block_table for req in out.decode],
        )
        for i, req in enumerate(out.decode):
            token = self.sampler.sample_one(
                logits[i], req.request_id, req.sampling_params
            )
            sampled[req.request_id] = token

    finished = self.scheduler.update_after_step(sampled, self.eos_token_ids)
    for req in finished:
        self.sampler.drop_request(req.request_id)
    return [req.request_id for req in finished]
```

**第 1 拍：点名。**
`self.scheduler.schedule()` 返回一个 `SchedulerOutput`：`out.prefill` 是本步
新准入的请求（刚上车），`out.decode` 是已经在跑的请求。调度器内部已经完成了
保守准入检查（块池装得下全程才放行）和块表的创建——这些 Week 5 都讲透了。

**第 2 拍：prefill 新人。**
对每个新请求：给提示词的**每个 token** 各领一个物理槽位（`append_slot`，
Week 4 的页表操作），然后把整段提示词一次性喂给 `model.prefill`。
`prefill` 返回**最后一个位置**的 logits（形状 `(V,)`），直接交给采样器采出
**第一个新词**。注意：这个新词也要记账，所以放进 `sampled` 字典。

**第 3 拍：批量 decode。**
对所有在跑的请求：每个请求先为自己的下一个 token 领一个槽位；然后取每个请求的
`last_token_id()`（它最后吃到的那一个词）拼成一个 `(B,)` 的张量，
`positions` 同理拼出每个新词的位置编号；一次 `model.decode_batch` 算出
`(B, V)` 的 logits——**B 个请求共享一次前向**，这就是批处理省时间的来源。
最后逐个采样，`logits[i]` 是第 `i` 个请求那一行。

**第 4 拍：登记下车。**
`update_after_step(sampled, self.eos_token_ids)` 把每个新词追加到对应请求的
`output_token_ids` 上，并检查两个下车条件：够长了（`"length"`）或撞停止符
（`"stop"`）。下车的请求会被调度器释放块、记到账，并以列表返回。

**第 5 拍（顺手）：清理采样器。**
`Sampler` 内部为每个带 seed 的请求维护了一个随机数发生器（Week 6），
请求下车后调用 `drop_request` 把它清掉——不下车不清，同请求多步采样才能
连着用同一个发生器。

最后返回**本步刚完成的 request_id 列表**——别小看这个返回值，第 3 章的
`stream` 就靠它判断"我这个请求做完了没"。

> 💡 **你可能会问：为什么 prefill 的请求当步就采出了第一个词，
> 却不在 `out.decode` 里？**
>
> 这是调度器定下的分工（Week 5）：刚准入的请求本步**只 prefill**——而 prefill
> 本身就会算出最后一个位置的 logits，顺手就能把第一个词采出来，白捡的。
> 它的**第二次**生成（也就是第一次真正的 decode）要等下一步心跳，
> 那时它已经"在跑"，会出现在 `out.decode` 里。所以你 trace 一个请求的完整生命周期：
> 第 1 步 prefill + 第 1 个词，第 2 步起每步 decode 一个词。

> ⚠️ **易踩坑：** `positions` 那一行的 `- 1` 特别容易写丢或写错。
> 顺序是：**先** `append_slot()` 给新 token 占位（此时 `num_tokens` 已经 +1），
> **再**算位置——所以新 token 的位置编号是 `num_tokens - 1`（位置从 0 数起）。
> 漏掉减 1，RoPE / 位置嵌入就把新词当成"下一个位置"的词，输出全乱，
> 而且测试不会报形状错误，只会静默地生成错误的词——对拍测试
> （`test_engine_generate_matches_single`）就是抓这个的。

## 2.4 一个细节：@torch.no_grad()

函数头上戴着 `@torch.no_grad()`。推理永远不需要算梯度（我们不训练），
关掉梯度记录能省内存、省时间。Week 1 的 `generate_naive` 上就有它，
这里只是延续惯例。**别漏了它**——漏了功能上也对，但 KV cache 之外会多攒一堆
无用的梯度图。

## 2.5 写完后怎么快速自检

`step()` 本身没有单独的测试用例（它是发动机，得装进车里试）——但你可以
先做个"手动摇引擎"的实验，不依赖 `generate`：

```python
# 接着第 1 章的 play 脚本
from minivllm.sampling.sampler import SamplingParams

rid = eng.add_request(tok.encode("a b c"), SamplingParams(max_new_tokens=3))
while eng.has_unfinished():
    done = eng.step()
    req = eng._requests[rid]
    print("本步下车:", done, "| 该请求已生成:", req.output_token_ids)
```

如果每一步 `output_token_ids` 都稳定变长一个、第三步后请求下车，
你的 `step()` 八九不离十就对了。之后第 3 章实现 `generate` 时，
`tests/test_w7.py` 的对拍测试会给出铁证。

> 📌 **对标 vLLM：** 真实 vLLM v1 里同样有一颗"一步"的心跳——
> `vllm/v1/engine/core.py` 里 `EngineCore.step()` 的 docstring 写得很直白：
> "Schedule, execute, and make output"，函数体依次是
> `scheduler.schedule()` → `execute_model` → `sample_tokens` →
> `scheduler.update_from_output(...)`，和我们这五拍一一对应。
> 区别主要是真实版把这个内核放进了独立进程（`EngineCore`），
> 输出处理（detokenize 等）在另一个进程里流水化——第 6 章会展开。

> 📌 **划重点：** `step()` = 点名 → prefill 新人（顺手采第一个词）→
> 批量 decode → 登记下车 + 清理采样器。它和 Week 5 循环体的差别只有两处：
> 采样器上岗、`while` 挪给调用方。

---

## 动手练习

1. 在 `minivllm/engine/llm_engine.py` 里实现 `LLMEngine.step`
   （照 2.3 节的五拍，先别看答案）；
2. 跑 2.5 节的"手动摇引擎"脚本，确认请求逐词变长、到点下车；
3. 此时 `generate` / `stream` 还没写，正式测试要等下一章。想提前对答案可以跑
   `IMPL=reference .venv/bin/pytest tests/test_w7.py -m "not slow"` 看终点长什么样。

## 参考答案

`reference/engine/llm_engine.py` 里的 `LLMEngine.step` 是标准答案，约 45 行。
**卡住 20 分钟再看**，重点对照三处：prefill 的新词有没有记入 `sampled`；
`positions` 是不是 `num_tokens - 1`；下车后有没有 `sampler.drop_request`。

---

👉 下一章：[第 3 章：generate 与 stream——两种用法](./03-generate与stream-两种用法.md)
