# vLLM 源码对照（Week 8 阅读材料）

本教程的每个 minivllm 模块都对应真实 vLLM 里的一处实现。
源码位置：pip 安装的 `vllm` 包内（`.venv/lib/python3.12/site-packages/vllm/`，
定位命令见 Week 1 第 1 章）。**版本锁定 v0.27.1**，书中行号即该版本的真实行号。

## 模块对照表

| minivllm | 真实 vLLM v1 | 它管什么 |
|---|---|---|
| `scheduler/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 调度器（continuous batching） |
| `scheduler/request.py` | `vllm/v1/request.py` | 请求的状态机 |
| `cache/block_pool.py` | `vllm/v1/core/block_pool.py` | 空闲块管理 |
| `cache/block_table.py` | `vllm/v1/worker/block_table.py` | 逻辑→物理块映射 |
| `cache/paged_cache.py` | `vllm/v1/attention/backends/` | 分页注意力的存取 |
| `sampling/sampler.py` | `vllm/v1/sample/` + `vllm/sampling_params.py` | 采样 |
| `engine/llm_engine.py` | `vllm/v1/engine/llm_engine.py` | 同步引擎 |
| `engine/llm.py` | `vllm/entrypoints/llm.py` | 用户门面 |
| `model/qwen3.py` | `vllm/model_executor/models/qwen3.py` | 模型实现 |

## 真实调度器 `schedule()` 的开篇注释（vllm/v1/core/sched/scheduler.py）

```python
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

和我们的 `Scheduler.schedule()` 对照着读：
真实版不区分"prefill 阶段 / decode 阶段"，统一成"每个请求该补算几个 token"
（这就是 chunked prefill 的视角）；还多了抢占（preemption）、前缀缓存、投机采样。
我们的版本把 prefill/decode 显式分开，是为了让第一次读的人看得懂。

## 真实块池（vllm/v1/core/block_pool.py）

真实版比我们的多了两件大事：

1. **前缀缓存（prefix caching）**：每个块算哈希，相同内容的块可以复用——
   两个请求有相同开头时，前面那段 KV 直接共享，不用重算；
2. **逐出（eviction）**：块不够时把暂时用不到的缓存块"降级"处理。

我们的 BlockPool 只有发块/收块，是去掉这两件大事后的骨架。

## 建议的带读顺序（Week 8 用）

1. 先读 `vllm/v1/request.py`：找 `RequestStatus`，和我们的一一对应；
2. 再读 `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 前 100 行：找
   waiting/running 队列、token budget，对照我们的保守准入；
3. 然后 `vllm/v1/core/block_pool.py` 的 `get_new_blocks()`：对照我们的 `allocate()`；
4. 最后 `vllm/model_executor/models/qwen3.py`：找 RMSNorm / RoPE / GQA /
   SwiGLU 四个关键词，每一处都是我们 Week 6 写过的。
