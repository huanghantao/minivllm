# minivllm：从零手写 LLM 推理引擎

> 参考真实 vLLM 源码，用 PyTorch 从零写一个**能在 Mac 上跑真实模型**的迷你推理引擎。

## 这是什么

这是一套 8 周的动手课程。你将从一个"每生成一个词就把整段话重算一遍"的朴素脚本出发，
一周一周地加上真正的优化——KV cache、分页（PagedAttention 的灵魂）、continuous batching、
采样器——最后组装成一台能加载 Qwen3-0.6B 真实权重、批量服务、流式输出的推理引擎，
并以此为地图，读懂真实 vLLM 的核心源码。

**你不需要**：懂 CUDA、懂操作系统、懂 Transformer、用过 PyTorch。这些课程里都会从零教。
**你需要**：会 Python（类、模块、pip），每周 7 小时以上。

## 仓库怎么组织

| 目录 | 是什么 |
|---|---|
| `learning-guide/weekN/` | 教程正文（按周阅读） |
| `minivllm/` | **你的战场**：函数体是 `TODO`，由你填写 |
| `reference/` | 参考答案：完整实现（卡住 20 分钟再看） |
| `tests/` | 测试：填对就变绿 |
| `figures/` | 生图脚本与教程配图 |
| `material/` | 真实 vLLM 源码阅读材料 |

## 三个常用命令

```bash
# 1. 见证终点：参考答案全绿
IMPL=reference .venv/bin/pytest tests

# 2. 看看战场：你的实现还没填，大片红
.venv/bin/pytest tests

# 3. 填完某周：只跑那周的测试
.venv/bin/pytest tests/test_w1.py
```

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

教程在 macOS（Apple Silicon）上开发验证（开发机具体配置见 Week 1 第 1 章 1.2 节），
模型用本地缓存的 Qwen3-0.6B。
需要加载真实权重的测试标记为 slow：`.venv/bin/pytest tests -m slow`。

## 课程地图

- **Week 1**：见证与上手——一个词一个词是怎么蹦出来的
- **Week 2**：手写 mini Transformer——引擎的心脏
- **Week 3**：KV cache——别重复算已经算过的
- **Week 4**：分页——向操作系统借智慧（PagedAttention 的灵魂）
- **Week 5**：调度器——continuous batching
- **Week 6**：采样与真实权重——让引擎说人话
- **Week 7**：引擎化——从脚本到服务
- **Week 8**：对标真实 vLLM——读懂源码
