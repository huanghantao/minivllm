# 第 3 章：高级特性导览——prefix caching 与 chunked prefill

> 本章你将：
> 1. 讲得清 **prefix caching（前缀缓存）** 解决什么浪费、靠什么实现（块哈希）；
> 2. 讲得清 **chunked prefill（分块预填充）** 解决什么卡顿、靠什么实现（token 预算）；
> 3. 能在真实源码里指出这两个特性的"案发现场"（精确到文件和函数）；
> 4. 顺带理解第 2 章见过的**抢占（preemption）**和它们的关系。

这一章不写代码，是纯导览——但它是你简历和面试里会被问到的东西，
而且有了前 7 周的底子，理解它们只需要一页纸。

---

## 3.1 prefix caching：大家的第一句话都一样

### 浪费在哪

想象一个客服机器人，每个请求的提示词开头都是同一段 2000 token 的
系统提示（"你是某某公司的客服……请遵守以下 30 条规则……"）。
100 个用户同时来，这 2000 个 token 的 prefill 就被**原样重算了 100 遍**——
注意力是无状态的，它不知道"这段话我上一个人刚算过"。

同样的场景还有：多轮对话（每一轮都带着前面所有轮的历史）、
Few-shot 提示（每个请求共享同样的示例段）。
**相同的前缀，KV 完全相同，重算纯属浪费。**

### 大概怎么做

分页给了我们一个现成的粒度：**块**。块大小固定（比如 16 个 token），
"内容相同的块"就可以复用。做法分三步：

1. **给每个装满的块算哈希。** 哈希的输入是"前一个块的哈希 + 本块的 token 序列"，
   链式传递，保证"内容相同且前文也相同"的块哈希相同。真实代码在
   `vllm/v1/core/kv_cache_utils.py` 第 576 行的 `hash_block_tokens`：

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. ..."""
```

2. **块写满后登记进一本"哈希 → 块"的字典。** 真实代码是
   `vllm/v1/core/block_pool.py` 第 225 行的 `cache_full_blocks`，
   字典叫 `cached_block_hash_to_block`；
3. **新请求准入前先查字典。** 就是第 2 章 2.5 节看到的那个钩子：
   调度器对 `num_computed_tokens == 0` 的新请求调
   `kv_cache_manager.get_computed_blocks(request)`
   （`vllm/v1/core/kv_cache_manager.py` 第 229 行），
   内部 `find_longest_cache_hit` 顺着块哈希找**最长连续命中**。
   命中 128 个 token？那这 128 个 token 的 KV 直接挂上现成的块，
   `num_computed_tokens` 从 128 起步，prefill 只算剩下的。

共享的块被多个请求指着——这就是为什么真实块有 `ref_cnt`（第 2 章 2.6 节），
也是为什么发新块前要 `_maybe_evict_cached_block` 抹掉旧登记（**逐出**：
块要给别人用了，它挂着的"我这有某段前缀的 KV"广告就得撤下来）。

### 我们差在哪

我们的 `BlockPool` 发块收块，没有哈希、没有共享——所以 minivllm 里两个
相同开头的请求会各算各的。骨架你都有了，前缀缓存 = 块池 + 哈希字典 + 引用计数。

> 📌 **对标 vLLM：** 前缀缓存三件套——
> 哈希：`vllm/v1/core/kv_cache_utils.py` 的 `hash_block_tokens`（第 576 行）；
> 登记：`vllm/v1/core/block_pool.py` 的 `cache_full_blocks`（第 225 行）；
> 查询：`vllm/v1/core/kv_cache_manager.py` 的 `get_computed_blocks`（第 229 行）。
> 调度器里的调用点在 `vllm/v1/core/sched/scheduler.py` 第 746 行附近。

## 3.2 chunked prefill：别让长提示词堵住所有人

### 卡顿在哪

回忆 Week 5 的甘特图：continuous batching 让请求随来随走，decode 阶段大家
挤在一个批次里，一步一个词，其乐融融。

但如果此时来了个 **8000 token 的超长提示词**呢？我们的引擎（和早期 vLLM）
会把 prefill 和 decode 分开：先一口气把这 8000 个 token 的 prefill 算完，
再回头继续大家的 decode。这口气可能要好几百毫秒——期间所有在跑的请求
**一个字都吐不出来**。用户体验上就是：某人上传了一篇长文档，
全群聊的打字机同时卡死半秒。

### 大概怎么做

第 2 章 2.3 节那段注释已经把答案给你了：真实调度器眼里没有"prefill 阶段 /
decode 阶段"，只有"每个请求还差几个 token 没算"。那么长 prefill 就可以
**切成好几步算**——这一步算 512 个，下一步再算 512 个，每一步里剩下的
预算照常分给 decode 的请求。

机制就是第 2 章见过的那两行（`vllm/v1/core/sched/scheduler.py` 第 517~524 行）：

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

- 该补的 token 数 = 总共该有的 − 已算的（对 prefill 中的请求，这个差值
  就是提示词剩下的部分；对 decode 中的请求，差值恒为 1）；
- `min(num_new_tokens, token_budget)` 一刀砍下：**本步预算就这么多，
  长 prefill 排队分几步领**。

效果：长提示词不再独占引擎，每一步都先保证 decode 的请求有词可吐，
剩余预算才喂给 prefill 的"切片"。**TTFT（首字延迟）和吞吐从此可以兼得**——
这是 vLLM v1 默认开启的特性。

### 我们差在哪

我们的 `Scheduler.schedule()` 里 prefill 是"一次吃饱"：`SchedulerOutput.prefill`
里的请求，当步就把整个提示词算完。要升级成 chunked prefill，
改动点你很熟悉：`schedule()` 里给 prefill 也发 token 预算、
模型的 `prefill` 接口支持"从中间位置继续算"、块表支持分段append——
全是见过的零件，但每一处都要动，所以我们把它留在"下一步"（第 6 章）。

> 💡 **你可能会问：prefix caching 和 chunked prefill 会打架吗？**
>
> 不但不打架，还是好搭档。前缀缓存命中后 `num_computed_tokens` 直接从
> 中间起步，chunked prefill 把"剩下的部分"切片算——在调度器眼里它们都是
> 同一个公式的参数：差值是多少、本步预算给多少。这就是那段
> `NOTE(woosuk)` 注释说的 "general enough to cover chunked prefills,
> prefix caching, speculative decoding"。

## 3.3 顺带的第三件：抢占

第 2 章已经见过案发现场（`scheduler.py` 第 590 行附近）：`allocate_slots`
发不出块时，从 running 队尾踢一个请求，收回它的块。被踢的请求状态变成
`PREEMPTED`，之后回到 waiting 重来。

把三个特性放在一起看，它们其实在回答同一个问题的三个侧面——
**"资源不够时怎么办"**：

| 特性 | 回答 |
|---|---|
| prefix caching | 少花：相同前缀的 KV 不重算、不多占 |
| chunked prefill | 匀着花：长 prefill 分期付款，别挤占别人的步 |
| preemption | 先欠着：块不够时请排在后面的先让位，回头补上 |

而我们的 minivllm 对同一个问题的回答是**保守准入**：装不下就不让进。
四个词——**省、匀、让、等**——就是调度器设计的全部心法。

> ⚠️ **易踩坑：** 别在面试里说"prefix caching 是缓存 token"。
> 缓存的是 **KV 块**（以块为粒度、按内容哈希），token 本身没有可缓存的
> 中间状态——这正是 Week 3"存 K/V 不存 Q"的延伸。

> 📌 **划重点：** prefix caching = 块哈希 + 哈希字典 + 引用计数（相同开头不重算）；
> chunked prefill = token 预算把长 prefill 切片（长提示词不堵别人）；
> 两者都挂在"num_computed_tokens 追 num_tokens_with_spec"这一个公式上。

---

## 动手练习

1. 在真实源码里找到前缀缓存的"查字典"现场并读 20 行上下文：

```bash
grep -n -A 20 "def get_computed_blocks" \
  $VLLM_SRC/vllm/v1/core/kv_cache_manager.py
```

   找到 `find_longest_cache_hit` 的调用，回答：为什么注释里强调
   "the computed blocks must be full"（命中的必须是装满的块）？
2. 用我们 Week 4 的概念算一笔账：系统提示 2000 token、块大小 16、
   100 个请求共享它。开了 prefix caching 后，这段系统提示在 KV cache 里
   占几个块？不开呢？（提示：2000 / 16 = 125 个装满的块。）
3. （思考题）chunked prefill 把 prefill 切片后，一个 8000 token 的提示词
   可能要 16 步才"prefill 完"。它的第一个输出 token 会比"一口气算完"
   更早还是更晚出现？那对**其他请求**的 TTFT 呢？

## 参考答案

本章没有要填的战场文件；答案都在真实源码与 `material/vllm-源码对照.md` 里。
第 1 题提示：没装满的块内容还会变，哈希会失效。第 3 题提示：对本人更晚
（要和 decode 分预算），对别人更早（不再被堵死）——这正是它存在的意义。

---

👉 下一章：[第 4 章：性能测量——TTFT 与吞吐](./04-性能测量-TTFT与吞吐.md)
