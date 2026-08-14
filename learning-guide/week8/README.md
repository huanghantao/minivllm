# Week 8 本周导读：对标真实 vLLM——读懂源码

先回头看看你手里有什么：一个能加载 Qwen3-0.6B 真实权重、带分页 KV cache、
continuous batching、采样、流式输出的推理引擎——**每一行都是你自己写的。**

这周是结业周，做三件事：

1. **对照**：把 minivllm 的每个模块，对到真实 vLLM v1 的源码文件上——你会发现
   那边没有一个概念是你没见过的；
2. **带读**：打开真实 `scheduler.py`，逐段读 `schedule()`。它比我们的大 20 倍，
   但骨架就是你 Week 5 写的那个；
3. **测量**：给引擎装上"体温计"（TTFT 和吞吐两个指标），然后拉上真实 vLLM
   同场实测，看看我们追到了几成功力、差距具体在哪。

这一周几乎不用写新算法，但它的分量不比任何一周轻：**能读懂工业级源码，**
**才是这 8 周真正的毕业证。**

## 本周章节

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：模块对照——你已经认识它们了](./01-模块对照-你已经认识它们了.md) | minivllm ↔ vLLM v1 模块对照表，逐行认领老朋友 |
| [第 2 章：带读 vLLM 调度器源码](./02-带读vLLM调度器源码.md) | 逐段读真实 `schedule()` 开篇、`RequestStatus`、`get_new_blocks` |
| [第 3 章：高级特性导览——prefix caching 与 chunked prefill](./03-高级特性导览-prefixcaching与chunkedprefill.md) | 两个我们没做的高级特性：解决什么、大概怎么做、源码在哪 |
| [第 4 章：性能测量——TTFT 与吞吐](./04-性能测量-TTFT与吞吐.md) | 亲手实现 `benchmark_throughput` 和 `measure_ttft` |
| [第 5 章：结业实测——我们追到了几成功力](./05-结业实测-我们追到了几成功力.md) | 同机同模型对战真实 vLLM，逐个分析差距来源 |
| [第 6 章：大结业——完整旅程回顾与下一步](./06-大结业-完整旅程回顾与下一步.md) | 8 周地图回顾；投机采样、量化、多卡……接下来学什么 |

## 学完你会得到什么

1. 打开真实 vLLM v1 的 `scheduler.py` / `block_pool.py` / `llm_engine.py` **不慌**——
   每个核心概念你都亲手写过迷你版；
2. 讲得清 prefix caching 和 chunked prefill 各自解决什么浪费、在源码的哪个角落；
3. 会给任何推理引擎量体温：TTFT（用户多久看到第一个字）和吞吐（引擎一秒吐多少词）；
4. 拿到一份同机实测的成绩单，并能一条条说出"我们和真实 vLLM 的差距来自哪里"；
5. 一张属于自己的下一步学习地图。

## 本周要填的战场文件

本周只有一份战场文件，在第 4 章填：

| 文件 | 要实现的函数 |
|---|---|
| `minivllm/bench/metrics.py` | `benchmark_throughput` / `measure_ttft` |

其余章节是"读"和"跑"：读真实 vLLM 源码（本地仓库
`~/codeDir/pythonCode/vllm`）、跑三个实测脚本。

## 常用命令速查

```bash
# 本周测试（第 4 章填完 metrics.py 后应变绿）
.venv/bin/pytest tests/test_w8.py

# 偷看终点：参考答案应该全绿
IMPL=reference .venv/bin/pytest tests/test_w8.py

# 实测三件套（第 5 章带读，每个都要几分钟）
.venv/bin/python scripts/bench_naive_hf.py      # naive vs HF 自带 cache
.venv/bin/python scripts/bench_minivllm.py      # 我们的引擎：b1 / b8 / TTFT

# 真实 vLLM 要用它自己的 venv（注意两个环境变量，原因见第 5 章）
HF_HUB_OFFLINE=1 VLLM_METAL_USE_PAGED_ATTENTION=0 \
  ~/codeDir/pythonCode/vllm-metal/.venv-vllm-metal/bin/python scripts/bench_vllm.py

# 把教程构建成书
make book
```

> 📌 **划重点：** Week 1~7 你造了一艘小船；Week 8 我们把它开到真船旁边，
> 上真船的甲板参观一圈，再用同一把尺子量一量两艘船。你会惊喜地发现：
> 真船上每一根栏杆，你都叫得出名字。
