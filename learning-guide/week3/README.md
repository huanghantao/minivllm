# Week 3 导读：KV cache——别重复算已经算过的

## 本周定位

Week 1 你用 `generate_naive` 让模型一个词一个词地蹦字，也亲眼看到它慢；
Week 2 你手写了 MiniTransformer，知道了每个词在每一层都要算出 Q、K、V 三样东西。

本周我们停下来问一个"斤斤计较"的问题：**每多生成一个词，就把整段话从头重算一遍，
这里面有多少是白算的？** 答案会让你心疼——绝大部分都是白算的。

而修法出奇地朴素：**算过的 K 和 V 存下来，下次直接用。** 这个"存下来"的东西就叫
**KV cache**，它是所有 LLM 推理引擎的第一块基石。vLLM 之后所有的花哨技术
（分页、调度、前缀缓存）全都是围着它转的。

> 📌 **划重点：** 本周不改变模型生成的**任何一个字**，只改变"算得多快、占多少内存"。
> 测试会证明：带 cache 和不带 cache，输出一字不差。

## 目录

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：浪费在哪——每步重算整段话](./01-浪费在哪-每步重算整段话.md) | 拿笔算一算 naive 生成到底白算了多少 |
| [第 2 章：KV cache——把中间结果存下来](./02-KVcache-把中间结果存下来.md) | K/V 是每个词的"档案"，实现 `NaiveKVCache.append_and_get` |
| [第 3 章：prefill 与 decode——一次吃饱与一口一口吃](./03-prefill与decode-一次吃饱与一口一口吃.md) | 实现 `generate_with_cache`，给 `forward` 接上 `use_cache` 路径 |
| [第 4 章：实测加速与内存账](./04-实测加速与内存账.md) | 实测快 3.0 倍；手算 Qwen3-0.6B 每 token 112 KB |
| [第 5 章：AI 联系——KV cache 是推理的头号内存](./05-AI联系-KVcache是推理的头号内存.md) | 为什么"能同时服务多少人"由 KV cache 决定 |

## 学完你会得到什么

1. 能一句话说清 KV cache 缓存的**是什么、为什么可以缓存**（同一个词的 K/V 不会变）；
2. 能手写 `NaiveKVCache` 和 `generate_with_cache`，并用对拍测试证明结果与 naive **一字不差**；
3. 能脱口而出 prefill / decode 两个阶段的区别——这是推理圈的黑话，也是 Week 5 调度器的地基；
4. 会用 `kv_cache_memory_bytes` 算内存账：给你模型尺寸和并发数，能估算 KV cache 要占多少内存；
5. `tests/test_w3.py` 全绿。

## 常用命令速查

```bash
# 跑本周全部测试（默认打你的 minivllm/ 战场）
.venv/bin/pytest tests/test_w3.py

# 卡住了，先看参考答案是不是全绿
IMPL=reference .venv/bin/pytest tests/test_w3.py

# 只跑某一个练习
.venv/bin/pytest tests/test_w3.py -k naive_cache      # 第 2 章
.venv/bin/pytest tests/test_w3.py -k memory_bytes     # 第 4 章
.venv/bin/pytest tests/test_w3.py -k matches_naive    # 第 3 章的对拍测试
```

> ⚠️ **易踩坑：** 本周要改的是 `minivllm/cache/kv_cache.py`、`minivllm/generate.py`，
> 以及 `minivllm/model/transformer.py` 里 Week 2 写过的 `forward`（要给它加上
> `past_kv` 分支）。**别动 `reference/`、`tests/`、`figures/` 里的任何东西。**
