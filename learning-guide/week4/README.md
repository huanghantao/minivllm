# Week 4 本周导读：分页——向操作系统借智慧

Week 3 我们用 KV cache 消灭了"重复计算"，生成速度快了 3 倍。但只要把 Week 3 的 `NaiveKVCache` 放到"很多请求同时来、长短不一、随时结束"的真实场景里，一个新的浪费就浮出来了：**内存被切得七零八碎，明明还有大把空闲，却塞不下新请求。**

这一周我们向操作系统借一个 60 年前的老智慧——**分页（paging）**，把 KV cache 切成固定大小的"块"，让每个请求的 K/V 散落在块里、用一张"页表"记住它们的位置。这就是 vLLM 的灵魂组件 **PagedAttention** 的存储层，也是全课程最重要的两周之一（另一周是 Week 5 的调度器）。

你完全不需要懂操作系统。我们会从"储物柜"讲起，把虚拟内存、页表这些听起来唬人的词全部翻译成生活语言。

## 本周章节

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：内存碎片——储物柜的智慧](./01-内存碎片-储物柜的智慧.md) | 连续分配为什么会留下"洞"；固定大小块 + 页表的分页思想 |
| [第 2 章：BlockPool 与 BlockTable——页表本尊](./02-BlockPool与BlockTable-页表本尊.md) | 亲手实现发块/收块的管理员和"逻辑→物理"翻译表 |
| [第 3 章：PagedKVCache——写进去，读回来](./03-PagedKVCache-写进去读回来.md) | 所有请求共享的分页仓库：`write` 按槽位写、`gather` 按槽位读 |
| [第 4 章：模型的分页接口——prefill 与 decode_batch](./04-模型的分页接口-prefill与decode_batch.md) | MiniTransformer 长出引擎接口，一批长短不一的请求一起走 |
| [第 5 章：铁证——分页和不分页一字不差](./05-铁证-分页和不分页一字不差.md) | 对拍实验：换存储方式，生成的词一个都不许变 |
| [第 6 章：AI 联系——PagedAttention 到底省了什么](./06-AI联系-PagedAttention到底省了什么.md) | 内存利用率如何变成吞吐量；我们和真实 vLLM kernel 的差别 |

## 学完你会得到什么

1. 讲得清"内存碎片"是什么、为什么分页能消灭它（用储物柜就能给外行讲明白）；
2. 亲手写出 `BlockPool`（块管理员）、`BlockTable`（页表）、`PagedKVCache`（分页仓库）三个类；
3. 让 MiniTransformer 学会 `prefill` / `decode_batch` 两个引擎接口，一批请求共享一个仓库；
4. 用一条对拍测试证明：**分页只是换了个存法，模型吐出的词和 Week 3 一字不差**；
5. 为 Week 5 的 continuous batching 备好地基——没有分页，调度器根本玩不转。

## 本周要填的战场文件

| 文件 | 要实现的函数 |
|---|---|
| `minivllm/cache/block_pool.py` | `BlockPool.allocate` / `BlockPool.free` |
| `minivllm/cache/block_table.py` | `BlockTable.append_slot` / `BlockTable.physical_slots` |
| `minivllm/cache/paged_cache.py` | `PagedKVCache.write` / `PagedKVCache.gather` |
| `minivllm/model/transformer.py` | `prefill` / `decode_batch`（模型、层、注意力三层的同名方法） |

## 常用命令速查

```bash
# 打自己的实现（战场，没填完时是红的）
.venv/bin/pytest tests/test_w4.py

# 只跑某一部分
.venv/bin/pytest tests/test_w4.py -k "pool or block_table"   # 第 2 章
.venv/bin/pytest tests/test_w4.py -k paged                   # 第 3 章
.venv/bin/pytest tests/test_w4.py -k paged_equals_naive      # 第 4、5 章

# 偷看终点：参考答案应该全绿
IMPL=reference .venv/bin/pytest tests/test_w4.py

# 把教程构建成书
make book
```

> 📌 **划重点：** Week 3 解决的是"别重复算"，Week 4 解决的是"别浪费内存"。两个加起来，才谈得上 Week 5 的"让一堆请求挤在一起跑"。
