# 第 6 章：AI 联系——离 vllm.LLM 还差什么

> 本章你将：
> 1. 盘一盘：我们的 `LLM` 和真实 `vllm.LLM` 之间，还差哪几块拼图；
> 2. 认识三样新东西的名字和分工：API server、AsyncLLM、多进程 EngineCore；
> 3. 在真实 vLLM 源码里给它们各自找到"门牌号"；
> 4. 带着一张"差距清单"进入 Week 8 的源码带读。

---

## 6.1 先庆祝，再清醒

上周结束时你手里是一堆零件；这一周结束时，你手里是一行代码就能起、
能批量、能流式的**引擎**，而且真能驱动 Qwen3-0.6B 说人话。
这确实是里程碑——值得庆祝三秒钟。

好，庆祝完了。现在打开 `.venv` 里安装的真实 vLLM 源码，
泼自己一盆冷水：我们的 `LLM` 大约 80 行，真实的 `vllm/entrypoints/llm.py`
近两千行；我们的引擎是**单机单进程同步阻塞**的，而真实 vLLM 要伺候的是
**成百上千个并发用户通过 HTTP 同时砸请求**。

差的不是算法（分页、调度、采样你都有了），差的是**"引擎外面那圈服务的壳"**。
本章把差距拆成三块拼图讲清楚。这一周不写代码——看懂差距，
下周（Week 8）才好带着地图去读真实源码。

## 6.2 拼图一：API server——从"函数调用"到"网络服务"

我们的引擎怎么被使用？**同一个 Python 进程里 import 进来，调方法**。
这有两个天然限制：

- 调用方必须是 Python（隔壁 Java 团队用不了）；
- 调用方必须和引擎在**同一台机器的同一个进程**里（手机 App、网页前端没戏）。

真实世界里，LLM 几乎总是以 **HTTP 服务**的形态出现：起一个服务器，
监听端口，客户端发 JSON 过来、收 JSON 回去——OpenAI 的 API 就是事实标准，
所以 vLLM 直接实现了**兼容 OpenAI 协议**的服务端：

```bash
vllm serve Qwen/Qwen3-0.6B     # 一行命令，引擎变成 HTTP 服务
```

之后任何语言、任何机器都能用：

```bash
curl http://localhost:8000/v1/chat/completions \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "你好"}]}'
```

这一层在真实仓库里的门牌号是 `vllm/entrypoints/openai/api_server.py`
（HTTP 骨架，基于 FastAPI）和 `vllm/entrypoints/openai/chat_completion/serving.py`、
`vllm/entrypoints/openai/completion/serving.py`
（把 OpenAI 协议的 JSON 翻译成引擎调用，再把引擎输出
翻译回 JSON——包括流式用的 SSE 逐片推送）。看出来了吗？**这些 "serving"
就是我们 `LLM` 门面干的活的网络化版本**：翻译 + 组装，只是翻译的对象从
"字符串 ↔ token id"升级成了"HTTP JSON ↔ 引擎调用"。

## 6.3 拼图二：AsyncLLM——别让一个人堵死一万人

假设真把 HTTP 服务架在我们的引擎上，马上会撞一堵墙：**我们的 `generate`
是同步阻塞的**——`while self.has_unfinished(): self.step()` 这个循环会把
整个线程占死，直到这一批全部做完。期间第二个用户的 HTTP 请求到了？
对不起，排队等着，连"收件"都办不到。

修法不是让引擎跑得更快，而是**换一套调度姿势**：事件循环 + 协程。
真实 vLLM 的答案在 `vllm/v1/engine/async_llm.py` 的 `AsyncLLM`：

- 每个客户端请求是一个**协程**（async 任务），不是一次阻塞调用；
- 引擎的 `step()` 被事件循环**周期性驱动**，两次心跳之间的空隙，
  事件循环去收新请求、给老请求推送流式增量；
- 于是"收件"和"开工"真正并行：一万个用户同时连着，每个人都在
  自己的节奏里收 token，谁也不堵谁。

我们的 `stream` 已经有了一点这个味道（生成器、`yield` 逐片吐），
但它是**同步**生成器——`for piece in stream(...)` 一样把调用方线程占死。
真实版对应的是 `async for` + 异步生成器。

> 💡 **你可能会问：我们的引擎能不能凑合包一层 asyncio？**
>
> 思路对，但有个硬骨头：`step()` 里的模型前向是**重 CPU/GPU 计算**，
> 协程不会让出控制权——`await` 一个 `step()` 期间，事件循环照样被堵。
> 所以真实 vLLM 干脆把计算挪去**别的进程**（下一块拼图），
> asyncio 只负责收发消息。异步 + 多进程是配套使用的。

## 6.4 拼图三：多进程——让计算和收发各干各的

打开 `vllm/v1/engine/` 目录，你会看到几个名字很说明问题的文件：

```
vllm/v1/engine/
├── llm_engine.py        ← 同步引擎（我们的 LLMEngine 对标它）
├── async_llm.py         ← 异步外壳（AsyncLLM）
├── core.py              ← EngineCore：真正干计算的"内核"
├── core_client.py       ← 和内核通信的客户端
├── detokenizer.py       ← 独立的 detokenize 进程
└── output_processor.py  ← 把内核输出整理成 RequestOutput
```

真实 vLLM v1 的架构是**多进程流水线**：

1. **前端进程**（API server / AsyncLLM）：收 HTTP 请求、tokenize、
   把请求丢进消息队列；
2. **EngineCore 进程**：独占 GPU，闷头跑"点名 → 前向 → 采样"的心跳
   ——就是我们 `step()` 里那一套，只是它住独立进程，谁也堵不到它；
3. **detokenizer / output processor**：增量 detokenize、
   包装 `RequestOutput`，再推回前端流给客户端。

进程之间用消息队列（ZMQ）通信。这么拆的好处正是 6.3 节那个硬骨头的解药：
**计算再重，也只是堵 EngineCore 自己的进程**；前端的事件循环永远轻盈，
随时能收新请求、推流式增量。

回头看我们的 minivllm：所有东西挤在一个进程里，tokenize、调度、前向、
采样、detokenize 全在 `step()` 一条线上——**教学上这是优点**（数据流向
一目了然），生产上这就是要拆的地方。

## 6.5 差距清单总表

| 能力 | minivllm（现在） | 真实 vLLM | 源码门牌号 |
|---|---|---|---|
| 分页 KV cache / 调度 / 采样 | ✅ 都有（教学简化版） | ✅ 生产级 | `vllm/v1/core/` |
| 批量 + 流式 | ✅ generate / stream | ✅ | `vllm/entrypoints/llm.py` |
| HTTP 服务（OpenAI 协议） | ❌ 只能 Python 内调用 | ✅ `vllm serve` | `vllm/entrypoints/openai/api_server.py` |
| 异步并发 | ❌ 同步阻塞 | ✅ AsyncLLM | `vllm/v1/engine/async_llm.py` |
| 多进程隔离 | ❌ 单进程 | ✅ EngineCore 独立进程 | `vllm/v1/engine/core.py` |
| 增量 detokenize | ⚠️ 全文 decode 减前缀 | ✅ 专门的增量解码器 | `vllm/v1/engine/detokenizer.py` |
| prefix caching / chunked prefill / 抢占 | ❌ | ✅ | Week 8 第 3 章导览 |

最后一行那三个词先混个脸熟就行——它们是 Week 8 高级特性导览的主角。

> 📌 **对标 vLLM：** 本周写完，你的 `minivllm/engine/llm_engine.py` 对标
> `vllm/v1/engine/llm_engine.py`（它的心跳大头其实已经挪进了 6.4 节的
> `EngineCore`），`minivllm/engine/llm.py` 对标
> `vllm/entrypoints/llm.py`——**同步单机这条主干，你已经和真实 vLLM
> 同构了**。差的全部在"外面那圈壳"：网络、异步、多进程。
> `material/vllm-源码对照.md` 里有完整的模块对照表，Week 8 会按它带读。

> 📌 **划重点：** 从"引擎"到"服务"差三块拼图：API server（网络化）、
> AsyncLLM（异步化）、多进程 EngineCore（隔离计算与收发）。
> 它们不改变引擎的算法，只改变"谁、在什么时候、以什么方式摇手柄"。

---

## 动手练习

本章是"联系与展望"章，练习以侦察为主：

1. （侦察题）打开 `.venv` 里的 `vllm/v1/engine/core.py`，
   找到 `EngineCore.step` 方法（docstring 就写着
   "Schedule, execute, and make output"），确认你能认出五拍里的至少三拍：
   点名（`schedule()`）、模型执行（`execute_model` / `sample_tokens`）、
   登记输出（`update_from_output`）；
2. （侦察题）打开 `vllm/entrypoints/openai/api_server.py`，搜 `chat_completions`，
   看看一个 HTTP 请求进来后是怎么被转手的——能找到它最终调用了哪个
   带 `generate` 字样的方法吗？
3. （思考题）6.3 节说"`await` 一个 `step()` 期间事件循环照样被堵"。
   如果不引入多进程，还有一个常见的 Python 手法能把重计算"挪出"事件循环——
   你能想到是什么吗？（提示：和"线程"有关；再想想为什么 vLLM 没选它，
   提示：GIL。）

## 参考答案

本章无代码答案。第 3 题提示：`asyncio.to_thread` / 线程池能把阻塞调用挪走，
但 Python 的 GIL 让多线程在 CPU 密集任务上无法真正并行——所以 vLLM 选了
多进程。第 1、2 题留到 Week 8 第 2 章的源码带读里对答案。

---

👉 下一周：[Week 8：对标真实 vLLM——读懂源码](../week8/README.md)
