# 第 5 章：AI 联系——这就是 vLLM 里的 model

> 本章你将：
> 1. 把 MiniTransformer 和真实 vLLM 的 `qwen3.py` 逐件对照——你写的每个类都有"真身"；
> 2. 搞懂一个关键认识：vLLM 的快，不在模型数学里，而在模型周围的工程里；
> 3. 看清 MiniTransformer 现在"慢在哪"——这就是 Week 3 要解决的重复计算问题。

---

## 5.1 打开真实 vLLM 的模型目录

真实 vLLM（v0.27.1，就装在你的 `.venv` 里）中，所有模型的实现都住在一个目录：

```
vllm/model_executor/models/     ← 每个模型一个文件：qwen3.py、llama.py、gpt2.py……
```

打开 `qwen3.py`，你会看到四个类——是不是眼熟得可怕？

| 你写的（minivllm） | 真实 vLLM（`qwen3.py`） | 干什么 |
|---|---|---|
| `MultiHeadAttention` | `Qwen3Attention`（约 65 行） | 注意力 |
| `MLP` | `Qwen3MLP`（从 `qwen2.py` 复用） | 前馈网络 |
| `DecoderBlock` | `Qwen3DecoderLayer`（约 173 行） | 一层 |
| `MiniTransformer` | `Qwen3Model` + `Qwen3ForCausalLM`（约 264、271 行） | 整机 |

点进 `Qwen3Attention` 看一眼，老朋友都在：

```python
self.qkv_proj = QKVParallelLinear(...)          # 我们的 q_proj / k_proj / v_proj 合并成一块
self.o_proj = RowParallelLinear(..., bias=False)  # 和我们的 o_proj 一样，连 bias=False 都一样
self.scaling = self.head_dim**-0.5               # 我们的 1/sqrt(d)
```

再看 `Qwen3DecoderLayer` 的 forward 骨架：`input_layernorm` → `self_attn` → 残差 →
`post_attention_layernorm` → `mlp` → 残差——和你写的 `DecoderBlock.forward`
是同一个节奏。

**这就是本周最大的收获：真实 vLLM 里的"model"，和你这周手写的东西是同构的。**
你不再是"大概知道 Transformer"，而是能打开工业级推理引擎的模型文件，认出每一个零件。

## 5.2 差别在哪？一张升级清单

当然，真家伙和我们的玩具之间有差距。把差别列出来，你会发现一个秘密：
**每一条差别，都是这门课后面某一周的主题。**

| 差别 | 我们（Week 2） | 真实 vLLM | 课程哪里讲 |
|---|---|---|---|
| KV cache | 没有，每步重算整段 | 每步只算新 token | **Week 3** |
| K/V 存哪 | 不存 | 分页块（PagedAttention） | **Week 4** |
| 批量请求 | 一次一句 | continuous batching | **Week 5** |
| 归一化 | LayerNorm | RMSNorm（更省） | **Week 6** |
| 位置编码 | 查表位置嵌入 | RoPE（旋转位置编码） | **Week 6** |
| 注意力头 | 每个头一份 K/V | GQA：多个 Q 头共享 K/V | **Week 6** |
| 权重 | 随机初始化 | 加载 Qwen3-0.6B 真实权重 | **Week 6** |
| 线性层 | `nn.Linear` | `QKVParallelLinear` 等（多卡切分） | 超出本课范围 |
| 打分/softmax | 朴素 PyTorch | 高度优化的底层 kernel | 超出本课范围 |

注意最后两行和前面的区别：**前面六行是"思想"上的差别，我们都会在 minivllm 里
亲手实现；最后两行是"工程"上的差别**（多卡并行、底层 kernel），是本课故意不碰的——
它们影响速度，不改变思想。

## 5.3 关键认识：vLLM 的快不在模型里

把这件事想透，你就抓住了这门课的题眼。

**模型数学（注意力、MLP、归一化）在 vLLM 里和在任何朴素实现里是一样的。**
`softmax(QKᵀ/√d)V` 不会因为写在 vLLM 里就变得更快——快的不是公式，是公式周围的
一整套工程：

1. **别重复算**：KV cache（Week 3）；
2. **内存别浪费**：分页管理（Week 4）；
3. **别让 GPU 闲着**：continuous batching（Week 5）；
4. **采样、流式、多请求**：引擎化（Week 6、7）。

一个类比：MiniTransformer 是一台**发动机**。Week 1 我们是把它抱在手里空转——
能转，但没装车。vLLM 的真正本事是围绕发动机造了整车：变速箱（调度器）、油箱管理
（分页 KV cache）、仪表盘（采样与输出）。发动机本身，你现在已经会造了。

这也解释了课程的定位：我们不碰 CUDA、不碰 kernel，因为那是"把车床造得更好"的学问；
我们学的是"这辆车为什么这么设计"——读完 vLLM 源码所需要的一切概念。

## 5.4 预告 Week 3：你的模型现在慢在哪？

最后，带着一个问题结束本周。回到 Week 1 的 `generate_naive`：每生成一个新词，
它都把**整段话**（提示词 + 已生成的所有词）重新丢给 `model(ids)` 算一遍。

代入你刚写完的代码想想这意味着什么。生成第 20 个词时：

- 前 19 个词的 Q、K、V **上周……不对，上一大步刚算过**，这次又原样算了一遍；
- 而且结果和上次**一模一样**（因果掩码保证了前面的词看不到后面，它们的表示不会变）；
- 真正"新"的计算，只有第 20 个词那一份。

生成一段话的时间大致随长度**平方增长**——越往后每步越贵，全在做重复的无用功。
真实数据（`figures/data/bench.json`，本机实测 Qwen3-0.6B）：这种"每步整段重算"的
naive 方式只有 **39.3 tok/s**，而真实 vLLM 能到 **145.8 tok/s**。

重复计算的部分是什么？正是每个词的 **K 和 V**——它们是"每个词留给后来者的档案"，
算一次就够，理应存下来。**怎么存、怎么取，就是 Week 3 的 KV cache。**
你在第 2 章埋的伏笔（`kv_len > q_len` 的掩码）和第 3 章留的接口（`past_kv`），
下周全部点亮。

> 📌 **对标 vLLM：** 本周对照的完整模块映射表在仓库的
> `material/vllm-源码对照.md`，Week 8 会按它逐个带读真实源码。
> 现在只需要建立信心：那个目录里的代码，你已经认识骨架了。

> 📌 **划重点：** 你手写的 MiniTransformer 和真实 vLLM 的 `qwen3.py` 同构——
> 注意力、层、整机一一对应。vLLM 的快不在模型数学，而在 KV cache、分页、
> 调度这些"模型周围的工程"，那正是接下来五周的内容。

## 动手练习

本章没有代码填空，练习是"去真实世界里认亲"：

1. 打开 `.venv` 里的 `vllm/model_executor/models/qwen3.py`，
   找到 `Qwen3Attention`、`Qwen3DecoderLayer`、`Qwen3ForCausalLM` 三个类，
   在你自己的 `minivllm/model/transformer.py` 里指出各自对应的类；
   在 `Qwen3Attention` 里找到 `self.scaling`，说出它对应你写的哪一行。
2. 再跑一遍本周全部测试，确认终点是绿的：

```bash
.venv/bin/pytest tests/test_w2.py
```

3. 思考题（下周揭晓答案）：`generate_naive` 生成第 100 个词时，
   整段 100 个词里只有几个词的 K/V 是"必须新算"的？为什么？

## 参考答案

本章没有代码填空。对照阅读的标准路径在 `material/vllm-源码对照.md`——
先看模块对照表即可，详细带读留给 Week 8。**卡住 20 分钟再看**的原则不变：
思考题先自己想，再想不通就翻下周的章节。

---

👉 下周见：Week 3《KV cache——别重复算已经算过的》，从
[浪费在哪——每步重算整段话](../week3/01-浪费在哪-每步重算整段话.md)开始。
