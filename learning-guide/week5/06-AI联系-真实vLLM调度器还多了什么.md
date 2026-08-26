# 第 6 章：AI 联系——真实 vLLM 调度器还多了什么

> 本章你将：
> 1. 把我们的 `Scheduler` 和真实 vLLM 的 `vllm/v1/core/sched/scheduler.py`
>    对上号——你已经认识它的骨架了；
> 2. 知道真实调度器多出来的三件大事：**token 预算、抢占、chunked prefill**，
>    各自解决什么问题；
> 3. 拿到一份带读地图（`material/vllm-源码对照.md`），为 Week 8 读源码热身。

---

## 6.1 先对号入座

本周你写的每个概念，在真实 vLLM 里都有同一个名字。打开 `.venv` 里安装的
vLLM 源码（定位方法见 Week 1 第 1 章），对照这张表：

| 你本周写的 | 真实 vLLM v1 | 备注 |
|---|---|---|
| `scheduler/request.py` 的 `Request` | `vllm/v1/request.py` | 状态机几乎一样，真实版多一个被抢占态 |
| `scheduler/scheduler.py` 的 `Scheduler` | `vllm/v1/core/sched/scheduler.py` | waiting/running 队列同名 |
| `schedule()` 点名 | 同名方法 `schedule()` | 真实版复杂得多，但开头就在做同样的事 |
| `update_after_step()` 收盘 | `update_from_output()` | 登记新词、判断结束、释放块 |
| 块池记账 | `vllm/v1/core/block_pool.py` | 真实版多块哈希（前缀缓存用） |

完整的模块对照（含读源码的建议顺序）在仓库的
[`material/vllm-源码对照.md`](../../material/vllm-源码对照.md)，Week 8
会正式用它，现在可以先翻一眼。

## 6.2 真实调度器的世界观：没有"两个阶段"，只有"欠几个 token"

真实 `schedule()` 的开篇有一段注释（`material/vllm-源码对照.md` 里完整收录），
值得逐句读：

> There's no "decoding phase" nor "prefill phase" in the scheduler.
> Each request just has the num_computed_tokens …
> At each step, the scheduler tries to assign tokens to the requests
> so that each request's num_computed_tokens can catch up its
> num_tokens_with_spec.

翻译过来：真实调度器眼里**根本没有"prefill 阶段"和"decode 阶段"两个名单**，
只有一件事——每个请求"已经算到第几个 token"（`num_computed_tokens`）落后
于"应该算到第几个"，每一步给各请求分配一些 token 配额，让它们追赶。

- 一个请求提示词 100 个词、一个都算过？这一步让它补 100 个——这就是 prefill；
- 一个请求只差最新 1 个？补 1 个——这就是 decode；
- 提示词 5000 个词、一步算不完？**这一步先补 2000 个，剩下的下步再补**——
  这就是 chunked prefill（6.5 节）。

我们的 minivllm 把 prefill/decode 显式分成两张名单，是为了让第一次读的人看得懂；
真实版把它统一成"补 token"的视角，是因为这个视角能一口气管住 chunked
prefill、前缀缓存、投机采样等所有花样。

## 6.3 多出来的第一件大事：token 预算

我们的准入卡点有两个：座位数（`max_num_seqs`）和块数。真实 vLLM 还多一个
更细的卡点：**每一步最多算多少个 token**（`max_num_batched_tokens`）。

为什么需要它？我们的一步里，decode 部分每个请求只算 1 个 token，工作量平稳；
但 prefill 部分一个请求就是几百上千个 token——如果一个步里同时准入好几个
长提示词请求，这一步会突然变得巨慢，所有在跑的请求都被卡住等它（首词延迟
和词间延迟全被拖累）。

token 预算就是给"一步的工作量"上个天花板：本步预算 2048 个 token，
长 prompt 用掉多少算多少，超了就把剩下的请求留到下一步。**这让每一步的耗时
变得均匀可预测**——流式输出时用户看到的打字机效果才平稳。

## 6.4 第二件大事：抢占（preemption）

回忆我们的"保守准入"：准入时就把全程的块记在账上，换来"中途绝不缺块"。
代价第 3 章说过——按**最大**生成长度记账太悲观，块被白白预订，能同时服务
的请求数偏少。

真实 vLLM 选择另一条路：**激进准入，按需分配，不够再说**。块池紧张时，
调度器会把 running 里的某个请求**抢占**——踢回 waiting 队列，块全部收回，
等宽裕了再重新进来。被抢占的请求有两种处理方式：

- **重算（recompute）**：KV cache 全扔掉，再进来时从头 prefill 一遍；
- **换出（swap）**：把 KV cache 临时搬到 CPU 内存，再进来时搬回来。

抢占让内存利用率拉满（不为"可能用不到"的块付定金），代价是调度器复杂度
暴涨：状态机多一个被抢占态、重新准入要恢复现场。我们第 2 章的状态机没有
回头箭头，正是因为保守准入把这个麻烦整个绕开了。

## 6.5 第三件大事：chunked prefill（切块预填）

我们第 4 章的分工是"prefill 逐个、decode 拼批"——长 prompt 会独自霸占一步。
chunked prefill 的做法是：**把长 prompt 切成块（比如每次 512 个 token），
和本步的 decode 混在同一个 batch 里算**。

这正是 6.2 节那个"补 token"世界观的直接产物：既然一切只是"补几个 token"，
那一个请求补 512 个（prefill 的一块）、另一些请求各补 1 个（decode），
完全可以拼在一步里。好处是长 prompt 不再堵路，词间延迟更平稳。

Week 8 第 3 章会有它和前缀缓存的专门导览，这里先混个脸熟。

## 6.6 本周回顾与下周预告

回头看，本周其实只讲了一句话：

> **每一步重新点名：做完的下车还块，排队的补位上车。**

但从这句话里长出了整个引擎的骨架：工单（`Request`）、状态机、两个队列、
准入记账、五工位流水线。加上 W3 的 KV cache、W4 的分页，你现在手里已经是
一个**五脏俱全的 proto-引擎**了。

它还差两样东西才能"说人话"：

1. **会采样**：现在只会贪心 argmax，不会 temperature、top-p 这些"个性旋钮"；
2. **认识真实模型**：MiniTransformer 是玩具，Qwen3-0.6B 的权重还没加载进来。

这就是 Week 6 的活：采样器 + 加载真实 Qwen3 权重。Week 7 再把
`continuous_batch_generate` 从函数重构成 `LLMEngine` 类，支持请求随时进
随时出、流式输出——那时它就真的是一个服务了。

> 📌 **对标 vLLM：** 真实调度器 = 我们的骨架 + token 预算
> （`max_num_batched_tokens`）+ 抢占（preemption）+ chunked prefill +
> 前缀缓存。每一样都能在 `material/vllm-源码对照.md` 里找到对应的源码位置，
> Week 8 我们拿着那张地图正式带读。

> 📌 **划重点：** 我们的保守准入用"记账"换来了简单；真实 vLLM 用 token 预算
> 控制每步工作量、用抢占换内存利用率、用 chunked prefill 削平长 prompt 的
> 尖峰。骨架你已经会了，剩下的都是权衡。

---

## 动手练习

1. （阅读练习）打开 `material/vllm-源码对照.md`，按"建议的带读顺序"的第 1、
   2 条，去真实仓库里找到 `vllm/v1/request.py` 的 `RequestStatus` 和
   `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 方法签名，截图或抄下
   和我们版本的三个不同点；
2. （思考题）我们的保守准入和真实 vLLM 的"激进准入 + 抢占"，各自适合什么
   场景？提示：想想"块池很大、请求很短"和"块池紧张、请求长短悬殊"两种情况；
3. 跑一遍本周全部测试，确认全绿收官：

```bash
.venv/bin/pytest tests/test_w5.py
```

## 参考答案

本章没有要填的代码。思考题参考要点：块池宽裕、请求短小时，保守准入的浪费
可以忽略，简单的优势大；块池紧张、请求长短悬殊时，保守记账会让大量块被
"可能用不到"的预订锁死，激进准入 + 抢占才能跑满——这也是真实服务选择后者的
原因。

---

👉 下一周：[Week 6：采样与真实权重——让引擎说人话](../week6/README.md)
