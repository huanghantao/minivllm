# Week 2 本周导读：手写 mini Transformer——引擎的心脏

Week 1 里，我们把模型当成一个黑盒：`generate_naive` 每次把一串 token id 丢给
`model(ids)`，拿回一堆分数，挑出下一个词。黑盒里面发生了什么？我们一次都没打开过。

**这周就打开它。** 你要亲手写出一个迷你 GPT——`MiniTransformer`：注意力、因果掩码、
多头、残差、LayerNorm、MLP、嵌入、输出头，一个零件不少。写完之后，Week 1 那个
`generate_naive` 不用改一行代码，就能把"你写的模型"跑起来——因为它俩本来就约好了
同一个接口：`forward(input_ids) -> logits`。

这就是本周的定位：**推理引擎的"心脏"长什么样，你自己造一颗。** 后面所有周
（KV cache、分页、调度、采样）都是围着这颗心脏做工程，心脏本身你得先认识。

## 本周目录

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：注意力——每个词都回头看看](./01-注意力-每个词都回头看看.md) | Q/K/V 各管什么；注意力四行数学；softmax 的直觉 |
| [第 2 章：因果掩码——不许偷看未来](./02-因果掩码-不许偷看未来.md) | `make_causal_mask`；手算 2×2；q_len≠kv_len 为 Week 3 埋点 |
| [第 3 章：多头与残差——把零件拼成一层](./03-多头与残差-把零件拼成一层.md) | 多头=多双眼睛；`_split_heads`/`_merge_heads`；LayerNorm/残差/MLP |
| [第 4 章：组装——MiniTransformer 诞生](./04-组装-MiniTransformer诞生.md) | 嵌入、堆叠 N 层、lm_head；整机跑通，8 个测试全绿 |
| [第 5 章：AI 联系——这就是 vLLM 里的 model](./05-AI联系-这就是vLLM里的model.md) | 对照真实 vLLM 的 `qwen3.py`；预告 Week 3 的重复计算问题 |

建议按顺序读：第 1、2 章搞定 `model/attention.py` 的两个函数，第 3、4 章搞定
`model/transformer.py` 的四个方法，第 5 章抬头看看真实世界。

## 学完你会得到什么

1. 一个**你亲手写的、能跑的最小 GPT**：输入 `(B, L)` 的 token id，输出
   `(B, L, vocab)` 的 logits，每个位置都在预测下一个词；
2. 对注意力机制的**肌肉记忆**：`softmax(QKᵀ/√d)V` 这四行，你能默写、能手算、
   能说出每一步的形状；
3. 看懂真实 vLLM 模型文件（`vllm/model_executor/models/qwen3.py`）的**基本盘**——
   你会发现那里的类名、结构和你写的几乎一一对应；
4. `tests/test_w2.py` 全绿，包括那条最有成就感的 `test_transformer_causality`：
   改掉未来的 token，过去位置的预测纹丝不动——因果性是你亲手焊死的。

## 常用命令速查

```bash
# 本周战场：minivllm/model/attention.py 和 minivllm/model/transformer.py

# 只跑注意力 + 掩码的 5 个测试（第 1、2 章收尾用）
.venv/bin/pytest tests/test_w2.py -k "mask or sdpa"

# 跑本周全部 8 个测试（第 4 章收尾用）
.venv/bin/pytest tests/test_w2.py

# 看看参考答案全绿的样子（任何时候想确认终点都可以跑）
IMPL=reference .venv/bin/pytest tests/test_w2.py
```

> 📌 **划重点：** 本周不写任何"推理引擎"的东西，只写模型。但请记住 Week 1 的
> 约定：`model(input_ids) -> logits` 这个接口不变——后面 6 周所有的加速，
> 都是在这个接口不变的前提下做的手术。
