# 第 4 章：组装流水线——continuous_batch_generate

> 本章你将：
> 1. 把前四周的零件——分页 KV cache（W4）、模型引擎接口（W4）、调度器（W5）——
>    第一次拧成一条完整的生成流水线；
> 2. 亲手实现 `continuous_batch_generate`：我们引擎的雏形（proto-引擎）；
> 3. 看懂主循环的五个工位：点名 → prefill → decode → 采样 → 登记；
> 4. 跑绿 `pytest tests/test_w5.py` 全部，包括"批处理结果和单跑一字不差"的铁证。

---

## 4.1 全局图：五个工位的流水线

本周的终极大活只有**一个函数**：`continuous_batch_generate`。给它一批提示词，
它内部开着调度器一步步点名，直到所有请求做完，把每个请求生成的词还给你。

主循环的每一步长这样（就是 Week 7 引擎 `step()` 的雏形）：

```
 while 还有没做完的请求:
     ① 点名    out = scheduler.schedule()          # 谁 prefill、谁 decode
     ② prefill 逐个跑新准入请求的整段提示词，各采一个词
     ③ decode  在跑的请求拼成一批，一步各采一个词
     ④ 采样    暂时就是 argmax（贪心）
     ⑤ 登记    scheduler.update_after_step(sampled) # 记账、请做完的下车
```

整个函数不到 50 行，但每一行都对应着前面某一周的血汗。咱们一段一段拼。

## 4.2 备料：块池、仓库、调度器、工单

```python
device = next(model.parameters()).device   # 模型在哪张卡上，数据就放哪
C = model.config
pool = BlockPool(num_blocks, block_size)
kv_cache = PagedKVCache(
    num_layers=C.num_layers,
    num_heads=C.num_heads,
    head_dim=C.head_dim,
    num_blocks=num_blocks,
    block_size=block_size,
    device=device,
)
scheduler = Scheduler(pool, max_num_seqs=max_num_seqs)
```

三件套都是老朋友：

- `BlockPool`：空块池子（W4）——**所有请求共用这一个池**；
- `PagedKVCache`：分页仓库（W4）——形状参数直接从 `model.config` 拿，
  我们的 `MiniConfig` 上正好有 `num_layers / num_heads / head_dim`；
- `Scheduler`：本章上一节的调度员，抱着池子上岗。

然后把提示词包成工单，全部塞进 waiting 队列：

```python
requests = []
for i, prompt in enumerate(prompts):
    params = SamplingParams(max_new_tokens=max_new_tokens)   # 统一参数，默认贪心
    req = Request(i, prompt, params)
    scheduler.add_request(req)
    requests.append(req)
```

注意 `requests` 这个列表**另外存了一份**所有工单（按 prompts 的顺序）——
因为请求做完就从 `running` 挪走了，最后要按原顺序交卷，得靠这份底稿。

## 4.3 prefill 工位：新客逐个点菜

```python
for req in out.prefill:
    slots = [req.block_table.append_slot() for _ in range(req.num_prompt_tokens)]
    ids = torch.tensor(req.prompt_token_ids, dtype=torch.long, device=device)
    logits = model.prefill(ids, kv_cache, slots, req.block_table)
    sampled[req.request_id] = int(torch.argmax(logits).item())
```

四行，每行都有出处：

1. **占座**：提示词有 L 个词，就 `append_slot()` L 次，拿到 L 个物理槽位号——
   还记得 Week 4 的分工吗：**占座是调度层的事**，模型只管算；
2. **备料**：提示词变成 `(L,)` 的张量，`dtype=torch.long` 是词表的索引类型；
3. **开算**：`model.prefill`（W4 第 4 章你写的）把整段提示词跑一遍，
   K/V 按 `slots` 写进仓库，返回最后一个位置的 logits `(vocab,)`；
4. **采样**：`torch.argmax(logits)` 取分数最高的词——`argmax` 就是"找最大值的
   下标"，`.item()` 把单元素张量变成 Python 整数。这就是贪心采样，Week 6 的
   `greedy_sample` 干的就是这件事。

采出的词记进 `sampled` 字典（键是 `request_id`），留给第 ⑤ 步登记。

> 💡 **你可能会问：为什么 prefill 是逐个跑，decode 却能拼成一批？**
>
> 因为各个请求的提示词**长度不同**，短的要补一大堆 0 才能和长的拼一起，
> 浪费太大（真实 vLLM 的 chunked prefill 有解法，第 6 章导览）。而 decode
> 每步每个请求**只出一个词**，天然一样长，拼批零浪费。这个"prefill 逐个、
> decode 拼批"的形状，和真实引擎是一致的。

## 4.4 decode 工位：拼成一批，一步一词

```python
if out.decode:
    slots = [req.block_table.append_slot() for req in out.decode]
    token_ids = torch.tensor(
        [req.last_token_id() for req in out.decode], dtype=torch.long, device=device)
    positions = torch.tensor(
        [req.block_table.num_tokens - 1 for req in out.decode],
        dtype=torch.long, device=device)
    logits = model.decode_batch(
        token_ids, positions, kv_cache, slots,
        [req.block_table for req in out.decode])
    for i, req in enumerate(out.decode):
        sampled[req.request_id] = int(torch.argmax(logits[i]).item())
```

逐步看：

1. **各占一个座**：decode 每步每个请求只新增一个词，所以每请求 `append_slot()`
   一次；
2. **本步输入**：`last_token_id()`——每个请求最新生成的那个词（第 2 章的属性，
   在这里派上用场）；
3. **位置编号**：`req.block_table.num_tokens - 1`。为什么减 1？`append_slot()`
   已经把新槽位记进块表，`num_tokens` 已经是"含新词"的总数，而新词的位置编号
   从 0 数起，就是总数减 1。**这一行必须写在 `append_slot()` 之后**；
4. **批量开算**：`model.decode_batch`（W4 第 4 章）返回 `(B, vocab)`，
   第 `i` 行就是第 `i` 个请求的下一个词分数；
5. **逐行采样**：`argmax(logits[i])`——**注意是对每一行分别 argmax**，
   不是对整个 `(B, vocab)` 张量 argmax（那会跨请求找出全场最高分，张冠李戴）。

> ⚠️ **易踩坑：** 三个高频错位——① `positions` 在 `append_slot()` **之前**
> 取了 `num_tokens`，位置全部差 1；② 对整个 `logits` 而不是 `logits[i]`
> 做 argmax；③ `sampled` 的键写成了列表下标 `i` 而不是 `req.request_id`——
> 调度器登记时按 `request_id` 查字典，键对不上就等于这步白跑。

## 4.5 登记与主循环

```python
while scheduler.has_unfinished():
    out = scheduler.schedule()
    sampled = {}
    # ……prefill 工位、decode 工位（如上）……
    scheduler.update_after_step(sampled, eos_token_ids)

return [req.output_token_ids for req in requests]
```

- 循环条件 `has_unfinished()`：waiting 或 running 非空就继续点名；
- 每步结束把 `sampled` 交给调度器收盘（上一章你写的 `update_after_step`）：
  登记新词、判断下车、还块销账；
- 全部做完后，按 `requests` 底稿的原始顺序，把每个请求的
  `output_token_ids` 交卷——**与 prompts 同序**，这是函数的合同。

整个函数再套一个 `@torch.no_grad()`（推理不算梯度，W4 的老规矩），收工。

## 4.6 两条铁证测试

`tests/test_w5.py` 的最后两条，是给整条流水线上的"保险"：

**`test_continuous_batching_matches_single`**：3 个长短不一的提示词
（`[3,1,4]`、`[1,5,9,2,6]`、`[2,7]`），`max_num_seqs=3` 一起跑批，
再逐个用 W4 的单请求函数 `paged_generate_single` 单独跑——**每个请求的结果
必须和单跑一字不差**。

这条测试的分量很重：批处理把请求拼来拼去、补 0、掩码、随时上下车，
任何一处串了数据，结果就会变。全对，才说明流水线是"透明"的——只改了速度，
没改行为。

**`test_continuous_batching_block_reuse`**：更狠——池子只给 2 块，
只装得下**一条**请求的全程（$3 + 4 = 7$ 个 token，$\lceil 7/4 \rceil = 2$ 块）。
此时保守准入会让第二条请求乖乖排队，等第一条下车、块还回来再进门：
**批处理自动退化成串行，结果依然正确**。这条测试同时验证了"还块销账"
真的发生了——否则第二条永远进不来，测试会卡死超时。

> 📌 **对标 vLLM：** 这个函数就是 `vllm/v1/engine/llm_engine.py` 里
> `step()` 的极简版：真实引擎的心跳也是"点名 → 执行 → 登记"三步，
> 只是执行部分换成了 GPU kernel 调度、采样换成了 Sampler。Week 7 我们会把
> 这个函数重构成 `LLMEngine` 类，长出长跑服务的能力。

> 📌 **划重点：** `continuous_batch_generate` = 三件套备料（池、仓库、调度器）
> + 五工位主循环（点名 → prefill → decode → 贪心采样 → 登记）。
> 每条请求一张工单、一张块表，块池和仓库全局共享。批处理只是排班，
> 结果必须与单跑一字不差。

---

## 动手练习

1. 填出 `minivllm/scheduler/batching.py` 里的 `continuous_batch_generate`
   （照 4.2–4.5 的五段，先自己写）；
2. 跑：

```bash
.venv/bin/pytest tests/test_w5.py
```

   本周全部测试应该变绿——包括两条铁证。
3. （选做，强烈建议）在循环里加一行
   `print([r.request_id for r in out.prefill], [r.request_id for r in out.decode])`，
   用 `max_num_seqs=2`、三个提示词跑一次，亲眼看"点名"是怎么一步步变化的——
   那就是第 1 章甘特图的下半张在你手里动起来。

## 参考答案

`reference/scheduler/batching.py`，整个函数不到 60 行。**卡住 20 分钟再看**，
重点对照：`positions` 的取值时机、`logits[i]` 的逐行 argmax、`sampled` 字典
的键是不是 `request_id`。

---

👉 下一章：[第 5 章：实测——批处理把吞吐抬高 3 倍](./05-实测-批处理把吞吐抬高3倍.md)
