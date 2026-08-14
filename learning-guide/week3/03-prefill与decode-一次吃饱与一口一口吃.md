# 第 3 章：prefill 与 decode——一次吃饱与一口一口吃

> 本章你将：
> 1. 认识推理圈最重要的两个黑话：**prefill**（预填充）和 **decode**（解码）；
> 2. 给 Week 2 写的 `forward` 接上 `past_kv` / `use_cache` 路径，重点是**位置编号要接着数**；
> 3. 逐行实现 `generate_with_cache`，并通过"与 naive 一字不差"的对拍测试；
> 4. 回收 Week 2 埋的伏笔：`make_causal_mask(q_len, kv_len)` 两个参数不相等的情形。

---

## 3.1 两个阶段：一次吃饱，然后一口一口吃

有了 KV cache，生成过程天然裂成两个阶段：

1. **prefill（预填充）**：把整段提示词**一次性**喂进模型。这一步算出每个词的
   K/V 存进 cache，顺带预测出**第一个新词**。像吃饭前先把满桌菜看一遍——一次吃饱；
2. **decode（解码）**：之后每一步，只把**刚生成的那一个词**喂进去。它的 Q 去查
   cache 里所有历史 K/V，预测出下一个词。一口一口吃，每口都很快。

```
提示词 [巴, 黎, 是]
   │
   ▼  prefill：一次算 3 个词，存 3 份 K/V，吐出第 1 个新词 "法"
   │
   ▼  decode：只喂 "法"  → 吐出 "国"
   ▼  decode：只喂 "国"  → 吐出 "首"
   ▼  decode：只喂 "首"  → 吐出 "都"
   ...
```

这两个词在推理圈子里天天出现，先给正式定义（以后看到不懵）：

| 术语 | 输入 | 每步产出 | 特点 |
|---|---|---|---|
| prefill | 整段提示词（长） | 第一个新词 + 全部 K/V | 一次大计算，**算得爽**（矩阵大、GPU 利用率高） |
| decode | 1 个新词 | 1 个新词 + 1 份新 K/V | 反复小计算，**等内存**（计算少，瓶颈在搬运权重） |

> 💡 **你可能会问：prefill 那次前向，和 naive 的第一步有什么区别？**
>
> 计算内容一模一样——都是整段从头算。区别在**善后**：naive 算完就把中间结果扔了，
> prefill 把每层的 K/V 存进了 cache，后面的步骤才有得用。所以 prefill 不省钱，
> 省钱的是 decode 阶段。整个流程是"一次性付清首付，之后每步只付零头"。

## 3.2 给 forward 接上 cache：三个升级点

打开 `minivllm/model/transformer.py`。Week 2 你写的 `MiniTransformer.forward`
签名是 `forward(self, input_ids)`，本周它升级成：

```python
def forward(self, input_ids, past_kv=None, use_cache=False):
    """朴素前向。input_ids: (B, L) 的 LongTensor。

    use_cache=False：返回 logits (B, L, vocab)；
    use_cache=True：返回 (logits, past_kv)，past_kv 是 NaiveKVCache。
    """
```

参考实现长这样，我们逐行拆：

```python
if use_cache and past_kv is None:
    past_kv = NaiveKVCache(self.config.num_layers)   # ① 第一次调用，新建空档案柜
offset = past_kv.seq_len() if past_kv is not None else 0  # ② 已经缓存了几个词？
B, L = input_ids.shape
positions = torch.arange(offset, offset + L, device=input_ids.device)  # ③ 位置接着数
x = self._embed(input_ids, positions)
for block in self.blocks:
    x = block(x, past_kv=past_kv)                    # ④ 把档案柜传给每一层
logits = self._head(x)
if use_cache:
    return logits, past_kv                           # ⑤ 连柜子一起还给调用方
return logits
```

**升级点 ①⑤ 都好懂：没柜子就建一个，用完连柜子还回去。**

**升级点 ②③ 是本周最重要的细节：位置编码要"接着数"。** 回忆 Week 2：每个词的
嵌入是"词嵌入 + 位置嵌入"，位置嵌入用 `pos_emb(positions)` 查表。decode 阶段
每次只进来 1 个新词，如果 `positions` 从 0 开始数，这个新词就会被当成
"序列里第 0 个词"——它的位置身份全错了，模型会一脸懵。正确的做法是
`offset = past_kv.seq_len()`：柜子里已经有 $S$ 个词了，新词的位置就是 $S$。

> ⚠️ **易踩坑：** 位置偏移忘加（`positions = torch.arange(L)` 一把梭），
> 是 KV cache 新手第一大 bug。它**不会报错**——形状全对，测试前两条可能都绿，
> 但对拍测试会挂：带 cache 的输出和 naive 对不上。遇到对不上，第一个查 positions。

**升级点 ④ 顺藤摸瓜到注意力层。** `DecoderBlock.forward` 把 `past_kv` 原样传给
`MultiHeadAttention.forward`，那里才是真正的存取现场：

```python
def forward(self, x, past_kv=None):
    """x: (B, L, hidden)。返回 (B, L, hidden)。"""
    q = self._split_heads(self.q_proj(x))
    k = self._split_heads(self.k_proj(x))
    v = self._split_heads(self.v_proj(x))
    if past_kv is not None:
        # 把本次的 K/V 存进仓库，并取出"含历史"的完整 K/V
        k, v = past_kv.append_and_get(self.layer_idx, k, v)
    mask = make_causal_mask(q.shape[2], k.shape[2]).to(x.device)
    out, _ = scaled_dot_product_attention(q, k, v, mask)
    return self.o_proj(self._merge_heads(out))
```

注意三行关键戏：

1. `q` 只用**本次输入**算——decode 时它就是 1 个词的 Q（形状 `(B, H, 1, D)`），
   印证了"Q 不缓存"；
2. `k, v = past_kv.append_and_get(...)`——存新的，拿回完整的；
3. `make_causal_mask(q.shape[2], k.shape[2])`——**两个参数不一样多了！**

## 3.3 回收伏笔：q_len ≠ kv_len 的掩码

Week 2 第 2 章我们留了个扣子：`make_causal_mask(q_len, kv_len=None)` 为什么
要两个长度？当时 $q\_len = kv\_len$，掩码是个方阵的下三角。现在 decode 阶段：

- $q\_len = 1$（本次只有 1 个新词在提问）；
- $kv\_len = S + 1$（cache 里 $S$ 个老词 + 它自己）。

掩码形状是 `(1, S+1)`，而且内容**全是 True**——这唯一的新词排在最后，
按因果律它可以看所有老词和自己。Week 2 手算过的"第 $i$ 行允许看前 $i+1$ 列"
在这里恰好是"第 0 行允许看全部 $S+1$ 列"。伏笔回收完毕：掩码机制一行没改，
自动适配了新场景。

> 💡 **你可能会问：decode 每步只喂 1 个词，logits 形状是 `(B, 1, V)`，
> 那 `logits[0, -1, :]` 还取"最后一个位置"吗？**
>
> 对，照样取。`logits[0, -1, :]` 的含义是"取本次输入最后一个位置的预测"——
> decode 时输入就 1 个词，最后一个位置就是它。这行代码 prefill/decode 通吃，
> 不用改。

## 3.4 实现 generate_with_cache

万事俱备，生成循环本身反而简单。合同在 `minivllm/generate.py` 的 docstring 里：

```python
@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens, eos_token_id=None):
    """带 KV cache 的生成：prefill 一次，之后每步只算新 token。"""
```

骨架（你来填血肉）：

```python
# ---- prefill：一次算完整个提示词，顺手把每层的 K/V 存进仓库 ----
logits, past_kv = model(input_ids, past_kv=None, use_cache=True)
next_token = torch.argmax(logits[0, -1, :]).item()
out_ids = [next_token]

# ---- decode：每步只把"刚生成的这一个 token"喂进去 ----
for _ in range(max_new_tokens - 1):
    if eos_token_id is not None and next_token == eos_token_id:
        break
    step_ids = torch.tensor([[next_token]], dtype=input_ids.dtype, device=input_ids.device)
    logits, past_kv = model(step_ids, past_kv=past_kv, use_cache=True)
    next_token = torch.argmax(logits[0, -1, :]).item()
    out_ids.append(next_token)

new_part = torch.tensor([out_ids], dtype=input_ids.dtype, device=input_ids.device)
return torch.cat([input_ids, new_part], dim=1)
```

三个容易翻车的地方，对照检查：

1. **prefill 白送了一个词。** prefill 的 logits 已经预测出第一个新词了，所以
   decode 循环只转 `max_new_tokens - 1` 圈。多转一圈就多生成一个词；
2. **eos 检查在"喂进去之前"。** 上一个词如果是 eos，直接 `break`，别再喂给模型。
   注意这个检查的位置保证：eos 本身**会**出现在输出里（它在上一轮就被 append 了），
   只是不再往后生成——这正是测试 `test_generate_with_cache_respects_eos` 钉死的行为；
3. **`past_kv` 要接力。** 每次调用都把上一次返回的 `past_kv` 传回去
   （`model(step_ids, past_kv=past_kv, ...)`），cache 才能一步步行进。
   传了 `None` 就等于每步都从零开始——那就退化成 naive 了。

## 3.5 铁证：对拍测试

本周最让人安心的一条测试：

```python
def test_generate_with_cache_matches_naive(mini_model):
    """带 cache 的生成必须和不带 cache 的逐 token 完全一致。"""
    ids = torch.tensor([[3, 1, 4]])
    naive = generate_mod.generate_naive(mini_model, ids, max_new_tokens=8)
    cached = generate_mod.generate_with_cache(mini_model, ids, max_new_tokens=8)
    assert naive.tolist() == cached.tolist()
```

`mini_model` 是测试夹具里一个**随机初始化**的 MiniTransformer（不用训练，
随机权重就够——我们比的是两条路径算得一样不一样，不是生成得好不好）。

这条测试绿了就说明一件事：**KV cache 是纯提速，不改语义。** 你发明的每一个
优化以后都要过这样的对拍关——快，且一个字都不能变。

> 📌 **对标 vLLM：** 真实 vLLM 的引擎主循环（`vllm/v1/engine/llm_engine.py` 一带）
> 每一步同样区分"这个请求是第一次来（prefill）还是已经在跑（decode）"，
> 只是它用调度器把很多请求的 prefill 和 decode 混在一步里执行——那是 Week 5 的故事。
> 本周你写的两段式循环，就是它的单请求简化版。

> 📌 **划重点：** prefill 一次吃饱（整段进、存全部 K/V、白送第一个词），
> decode 一口一口吃（每步只进 1 个词、位置接着数、K/V 接力传递）。
> 判卷标准：与 naive 输出**一字不差**。

---

## 动手练习

1. 升级 `minivllm/model/transformer.py`：`MiniTransformer.forward` 按 3.2 节加上
   `past_kv` / `use_cache` 逻辑（`positions` 的 offset 是灵魂）；
   `DecoderBlock.forward` 和 `MultiHeadAttention.forward` 把 `past_kv` 传下去、
   在注意力里调 `append_and_get`；
2. 实现 `minivllm/generate.py` 里的 `generate_with_cache`（3.4 节）；
3. 跑：

```bash
.venv/bin/pytest tests/test_w3.py
```

   重点看 `test_generate_with_cache_matches_naive` 绿不绿。如果对不上，
   按 3.2 节的 ⚠️ 先查 `positions` 的 offset，再查 `past_kv` 有没有接力传递。

## 参考答案

`reference/model/transformer.py`（`forward` 与 `MultiHeadAttention.forward`）和
`reference/generate.py`（`generate_with_cache`）是标准答案。**卡住 20 分钟再看。**
看的时候重点对照三件事：位置偏移、eos 检查的位置、循环转几圈。

---

👉 下一章：[第 4 章：实测加速与内存账](./04-实测加速与内存账.md)
