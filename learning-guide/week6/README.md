# Week 6 导读：采样与真实权重——让引擎说人话

## 本周定位

先承认一个事实：到目前为止，我们的引擎还是个"口吃"——

前五周它生成文字永远用**贪心**（每步挑分数最高的词），而且跑的模型是我们
Week 2 手捏的教学版 MiniTransformer：随机初始化的权重，说出来的话是乱码。

本周一次解决这两个"不像人"的地方：

1. **前半周（第 1–3 章）：采样。** 模型的输出其实是 15 万个分数，
   "怎么从分数里挑下一个词"本身就是一门学问：贪心、温度、top-k、top-p、
   随机种子。学完你就看懂了聊天界面上"温度"那个滑块到底在拧什么。
2. **后半周（第 4–6 章）：换一颗真的心脏。** 把 MiniTransformer 升级成
   真实的 Qwen3-0.6B——只差四处改动（RMSNorm、RoPE、GQA、SwiGLU），
   然后把 HuggingFace 上 1.4 GB 的真实权重灌进来，和官方实现**逐词对拍**。
   通过后，你的引擎说的每一个字都和 HuggingFace 一模一样。

> 📌 **划重点：** 本周是全课程的"见证奇迹"时刻——你 Week 2 学的每个零件
> （注意力、归一化、MLP、残差）会原封不动地出现在一个真实工业模型里，
> 只是各做了一个小升级。你会发现：真实模型并不神秘。

## 目录

| 章节 | 一句话内容 |
|---|---|
| [第 1 章：从 logits 到词——greedy 与 temperature](./01-从logits到词-greedy与temperature.md) | softmax 把分数变概率；贪心与温度缩放，实现 `greedy_sample` / `apply_temperature` |
| [第 2 章：top-k 与 top-p——给随机性画个圈](./02-topk与topp-给随机性画个圈.md) | 用 `-inf` 把不想要的词踢出局，实现 `top_k_filter` / `top_p_filter` |
| [第 3 章：Sampler——每个请求一套参数](./03-Sampler-每个请求一套参数.md) | `SamplingParams` 参数表、`sample_token` 流水线、seed 复现 |
| [第 4 章：Qwen3 的四处升级——RMSNorm 与 SwiGLU](./04-Qwen3的四处升级-RMSNorm与SwiGLU.md) | 归一化和 MLP 的两处升级，实现 `RMSNorm.forward` / `Qwen3MLP.forward` |
| [第 5 章：RoPE 与 GQA——位置编码与省内存](./05-RoPE与GQA-位置编码与省内存.md) | 旋转位置编码的最朴素讲法；KV 头减半的 GQA |
| [第 6 章：加载真实权重——和 HuggingFace 对拍](./06-加载真实权重-和HuggingFace对拍.md) | `from_pretrained` 逐行讲；三条对拍测试钉死"一字不差" |
| [第 7 章：AI 联系——模型文件里都装了什么](./07-AI联系-模型文件里都装了什么.md) | safetensors / config.json / tokenizer 三件套巡礼 |

## 学完你会得到什么

1. 能用一句话讲清 temperature / top-k / top-p 各自在拧什么旋钮，
   并手写完整的采样流水线；
2. 说得清 Qwen3 相对"教科书 Transformer"的四处升级各自解决什么问题；
3. 亲手让引擎加载 Qwen3-0.6B 真实权重，**logits 与 HuggingFace 偏差 < 5e-3、
   贪心生成逐词一致**——这是本周的金牌证书；
4. 打开任何一个 HuggingFace 模型目录，认得里面每个文件是干什么的；
5. `tests/test_w6_sampler.py` 全绿，`tests/test_w6_qwen3.py -m slow` 三条对拍全绿。

## 常用命令速查

```bash
# 前半周：采样器（快测，秒级）
.venv/bin/pytest tests/test_w6_sampler.py
.venv/bin/pytest tests/test_w6_sampler.py -k "greedy or temperature"   # 第 1 章
.venv/bin/pytest tests/test_w6_sampler.py -k "top"                     # 第 2 章

# 后半周：真实权重对拍（慢测，要加载 1.4 GB 权重，几十秒）
.venv/bin/pytest tests/test_w6_qwen3.py -m slow

# 卡住了，先看参考答案是不是全绿（见证终点）
IMPL=reference .venv/bin/pytest tests/test_w6_sampler.py
IMPL=reference .venv/bin/pytest tests/test_w6_qwen3.py -m slow
```

> ⚠️ **易踩坑：** 本周要改的是 `minivllm/sampling/sampler.py` 和
> `minivllm/model/qwen3.py`。**别动 `reference/`、`tests/`、`figures/` 里的
> 任何东西。** 慢测要读本地 HF 缓存里的权重，如果网络抽风可以加
> `HF_HUB_OFFLINE=1` 强制离线。
