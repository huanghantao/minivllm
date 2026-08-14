# Week 7 导读：引擎化——从脚本到服务

## 本周定位

先盘点一下你手里的家当：

- Week 4：分页 KV cache，模型学会了 `prefill` / `decode_batch` 两个引擎接口；
- Week 5：调度器 + `continuous_batch_generate`，一批请求随到随走、随走随上；
- Week 6：采样器（每个请求一套采样参数）+ 能加载 Qwen3-0.6B 真实权重的模型。

零件全齐了。但 Week 5 把它们组装成的是一个**函数**：`continuous_batch_generate`
一调用，请求一次给齐、结果一次拿全，函数返回，一切结束。这就像一台"只能煮一锅饭"
的电饭煲——米必须一次放好，煮好了才能开盖。

本周我们做最后一步组装：**把这个函数拆开来、立起来，变成一台长跑的引擎
`LLMEngine`**。它有心跳（`step`）、有收件箱（`add_request`）、有两种取件方式
（`generate` 一次拿全 / `stream` 逐字吐出）。最后我们再给它包一层门面 `LLM`，
做到和 vLLM 一模一样的用法：

```python
llm = LLM(model_name="Qwen/Qwen3-0.6B")
outs = llm.generate(["你好"], SamplingParams(max_new_tokens=32))
```

> 📌 **划重点：** 本周几乎不引入新算法——零件你都写过了。本周的核心是
> **"组织方式"**：把一次性流水线变成一台可以随时进件、持续运转、流式出件的机器。
> 这正是"推理引擎"和"推理脚本"的分界线。

## 目录

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：从函数到引擎——为什么要 LLMEngine](./01-从函数到引擎-为什么要LLMEngine.md) | Week 5 那个函数的三个局限；引擎架构全景图 |
| [第 2 章：step——引擎的心跳](./02-step-引擎的心跳.md) | 实现 `step()`：点名 → prefill 新人 → 批量 decode → 采样 → 登记下车 |
| [第 3 章：generate 与 stream——两种用法](./03-generate与stream-两种用法.md) | 同步批生成；流式的增量文本算法（decode 全文再减前缀） |
| [第 4 章：LLM 门面——一行起引擎](./04-LLM门面-一行起引擎.md) | 逐行读 `LLM.__init__`；实现 `_best_device` 和两个包装方法 |
| [第 5 章：实战——多人同时聊天](./05-实战-多人同时聊天.md) | 用真实 Qwen3 批量 + 流式生成；跑绿 `tests/test_w7.py`（含 slow） |
| [第 6 章：AI 联系——离 vllm.LLM 还差什么](./06-AI联系-离vllmLLM还差什么.md) | API server、AsyncLLM、多进程——从"引擎"到"服务"的最后一段路 |

## 学完你会得到什么

1. 能脱口而出 `LLMEngine.step()` 的五拍：点名 → prefill 新请求 → 批量 decode →
   采样 → 登记下车；
2. 能手写 `generate`（同步批）和 `stream`（增量文本 = 全文 decode 减去已发前缀）；
3. 能说清 `LLM` 门面每一行在干什么：定位权重、选设备、配 KV cache 容量、组装引擎；
4. 能用自己写的引擎让真实 Qwen3-0.6B 同时伺候多个用户，还能逐字流式输出；
5. `tests/test_w7.py` 快测全绿，slow 测试（真实模型）也全绿。

## 常用命令速查

```bash
# 快测（MiniTransformer + ToyTokenizer，秒级）
.venv/bin/pytest tests/test_w7.py -m "not slow"

# 卡住了，先看参考答案是不是全绿
IMPL=reference .venv/bin/pytest tests/test_w7.py -m "not slow"

# 只跑某一块
.venv/bin/pytest tests/test_w7.py -k stream      # 第 3 章的流式测试
.venv/bin/pytest tests/test_w7.py -k stop_token  # 停止符测试

# 真实模型端到端（要加载 Qwen3-0.6B 权重，较慢；离线可加 HF_HUB_OFFLINE=1）
.venv/bin/pytest tests/test_w7.py -m slow
```

（注意：`pytest tests/test_w7.py` 不加 `-m` 会把快测和 slow 一起跑。）

> ⚠️ **易踩坑：** 本周要填的是 `minivllm/engine/llm_engine.py` 里的
> `step` / `generate` / `stream`，以及 `minivllm/engine/llm.py` 里的
> `_best_device` / `generate` / `stream`。**别动 `reference/`、`tests/`、
> `figures/` 里的任何东西。**
