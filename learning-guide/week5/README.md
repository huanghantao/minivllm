# Week 5 导读：调度器——continuous batching

## 本周定位

先盘点一下你手里的牌：

- Week 3：KV cache 让**一个请求**不重复算；
- Week 4：分页 KV cache 让**一堆请求**共享一个大仓库，各自的块表记各自的位置，
  模型还长出了 `prefill` / `decode_batch` 两个引擎接口。

万事俱备，只差一个**调度员**：请求陆续进门，谁先算、谁后算、做完了谁下车、
块还回池子给谁用——这就是本周的主角 **Scheduler**，以及它实现的
**continuous batching（连续批处理）**。

这是全课程最重要的两周之一（另一周是 Week 4）。continuous batching 是 vLLM
赖以成名的核心技术：它不改模型、不改输出，只改"排班表"，就能把吞吐抬高 3 倍
（第 5 章实测）。本周结束时，你的 minivllm 将第一次具备"引擎"的完整形状——
Week 7 只是把它从函数包装成类。

> 📌 **划重点：** 本周依然不改变模型生成的**任何一个字**。测试会证明：
> 8 个请求一起跑，和每个请求单独跑，结果一字不差。我们改的只有"快"。

## 目录

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：静态批处理——陪跑的浪费](./01-静态批处理-陪跑的浪费.md) | 甘特图讲故事：一锅同时出锅，短请求全程陪跑 |
| [第 2 章：Request 与两个队列——waiting 与 running](./02-Request与两个队列-waiting与running.md) | 请求的三态状态机；两种 finish_reason |
| [第 3 章：Scheduler——每步点名](./03-Scheduler-每步点名.md) | 实现 `schedule` / `update_after_step`；保守准入与 `_reserved_blocks` 记账 |
| [第 4 章：组装流水线——continuous_batch_generate](./04-组装流水线-continuous_batch_generate.md) | 把前四周的零件拧成一条完整的生成流水线 |
| [第 5 章：实测——批处理把吞吐抬高 3 倍](./05-实测-批处理把吞吐抬高3倍.md) | 19.6 → 62.3 tok/s，为什么 b8 提升明显 |
| [第 6 章：AI 联系——真实 vLLM 调度器还多了什么](./06-AI联系-真实vLLM调度器还多了什么.md) | token 预算、抢占、chunked prefill，各一段 |

## 学完你会得到什么

1. 能用甘特图讲清静态批处理的"陪跑"浪费，以及 continuous batching 怎么消灭它；
2. 能手写 `Scheduler` 的三个核心方法，说清"保守准入"为什么不需要抢占；
3. 能手写 `continuous_batch_generate`，把调度器、分页 KV cache、模型引擎接口
   组装成一条流水线——**这就是引擎的雏形**；
4. 能解释"批处理为什么提吞吐而不降延迟"，并引用实测数字；
5. 打开真实 vLLM 的 `vllm/v1/core/sched/scheduler.py` 时，waiting/running 队列、
   准入判断这些概念全都眼熟；
6. `tests/test_w5.py` 全绿。

## 常用命令速查

```bash
# 跑本周全部测试（默认打你的 minivllm/ 战场）
.venv/bin/pytest tests/test_w5.py

# 卡住了，先看参考答案是不是全绿
IMPL=reference .venv/bin/pytest tests/test_w5.py

# 只跑某一章的练习
.venv/bin/pytest tests/test_w5.py -k request     # 第 2 章
.venv/bin/pytest tests/test_w5.py -k scheduler   # 第 3 章
.venv/bin/pytest tests/test_w5.py -k continuous  # 第 4 章
```

> ⚠️ **易踩坑：** 本周要填的是 `minivllm/scheduler/scheduler.py` 里的三个方法
> （`_can_admit` / `schedule` / `update_after_step`）和
> `minivllm/scheduler/batching.py` 里的 `continuous_batch_generate`。
> `Request` 本周已完整给出，直接读、直接用。
> **别动 `reference/`、`tests/`、`figures/` 里的任何东西。**
