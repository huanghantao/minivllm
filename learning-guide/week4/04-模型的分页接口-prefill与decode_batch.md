# 第 4 章：模型的分页接口——prefill 与 decode_batch

> 本章你将：
> 1. 明白模型为什么要"长出"两个新接口，而不是继续用 Week 3 的 `forward`；
> 2. 实现 `prefill`：一个请求的整段提示词，写进分页仓库再算注意力；
> 3. 实现 `decode_batch`：**一批长短不一的请求**各生成一个 token，共享同一个仓库；
> 4. 搞清楚 `slot_mapping` 和 `block_table` 是谁负责维护的（不是模型！）。

---

## 4.1 为什么 forward 不够用了

Week 3 的 `forward(input_ids, past_kv, use_cache=True)` 有个隐含假设：**一次只服务一个请求**——`past_kv` 是某个请求专属的 `NaiveKVCache`，里面就是这个请求自己的 K/V。

引擎的世界不是这样的：

- 仓库**全局只有一个**（`PagedKVCache`），所有请求共用；
- 模型算完 K/V，得有人告诉它"**写到哪几个槽位**"——这就是 `slot_mapping`；
- 算注意力前，得有人告诉它"**这个请求的历史在哪些槽位**"——这就是 `block_table`。

所以模型需要两个新接口，签名里多出这些"仓库通行证"：

```python
prefill(input_ids, kv_cache, slot_mapping, block_table)      # 一个请求的提示词
decode_batch(token_ids, positions, kv_cache, slot_mapping, block_tables)  # 一批请求各一步
```

先记住分工，这很重要：

| 角色 | 负责什么 |
|---|---|
| **调度层**（本周的测试代码、Week 5 的 Scheduler） | 维护块表：调 `append_slot()` 占座、把槽位号收集成 `slot_mapping` 传给模型 |
| **模型**（本章实现的 `prefill` / `decode_batch`） | 只管算：按 `slot_mapping` 写仓库、按 `block_table` 读历史、做注意力 |

模型**绝不**自己调用 `append_slot`——它不知道"该不该扩容、池子还够不够"，那是调度的事。这个边界画清楚了，Week 5 的调度器才能插进来。

## 4.2 prefill：一次吃饱（分页版）

先回顾 Week 3 的 prefill：把整段提示词一次性喂进模型，每个位置的 K/V 都算出来存好。分页版做的事一样，只是"存"的方式变了。模型的 `prefill` 入口很薄：

```python
@torch.no_grad()
def prefill(self, input_ids, kv_cache, slot_mapping, block_table):
    """一个请求的 prefill。input_ids: (L,)。返回最后一个位置的 logits (vocab,)。"""
    L = input_ids.shape[0]
    positions = torch.arange(L, device=input_ids.device)
    x = self._embed(input_ids, positions)
    for block in self.blocks:
        x = block.prefill(x, kv_cache, slot_mapping, block_table)
    return self._head(x[-1])
```

- `input_ids` 形状 `(L,)`——注意**没有 batch 维**了，因为 prefill 一次只处理一个请求；
- `torch.arange(L)` 生成位置编号 0, 1, …, L-1，喂给位置嵌入（Week 2 讲过：位置嵌入告诉模型"你是第几个词"）；
- `@torch.no_grad()`：推理不需要算梯度（反向传播才用），关掉省内存省时间；
- 最后只取 `x[-1]`（最后一个位置的隐藏状态）过输出头——Week 3 的经验：**只有最后一个位置的预测才是"下一个词"**，所以返回形状 `(vocab,)`。

`DecoderBlock.prefill` 只是把 Week 2 的老配方换个方法名（LayerNorm → 注意力 → 残差 → LayerNorm → MLP → 残差），真正的动作全在 `MultiHeadAttention.prefill`：

```python
def prefill(self, x, kv_cache, slot_mapping, block_table):
    """处理一个请求的整段提示词。x: (L, hidden)。返回 (L, hidden)。"""
    L = x.shape[0]
    q = self.q_proj(x).view(L, self.num_heads, self.head_dim)  # (L, H, D)
    k = self.k_proj(x).view(L, self.num_heads, self.head_dim)
    v = self.v_proj(x).view(L, self.num_heads, self.head_dim)

    kv_cache.write(self.layer_idx, slot_mapping, k, v)
    slots = block_table.physical_slots()  # 该请求全部 token 的物理槽位
    k_all, v_all = kv_cache.gather(self.layer_idx, slots)  # (L, H, D)

    q4 = q.permute(1, 0, 2).unsqueeze(0)  # (1, H, L, D)
    k4 = k_all.permute(1, 0, 2).unsqueeze(0)
    v4 = v_all.permute(1, 0, 2).unsqueeze(0)
    mask = make_causal_mask(L, L).to(x.device)
    out, _ = scaled_dot_product_attention(q4, k4, v4, mask)
    out = out.squeeze(0).permute(1, 0, 2).reshape(L, -1)  # (L, H*D)
    return self.o_proj(out)
```

按"三步走"读：

**第 1 步：算 K/V，写进仓库。** 三个投影各自 `.view(L, H, D)`（把 hidden 拆成 H 个头，每头 D 维，Week 2 的 `_split_heads` 同款，只是没有 batch 维）。然后 `kv_cache.write(...)`——刚算出的 L 个 token 的 K/V，按调度层给的 `slot_mapping` 写进各自槽位。

**第 2 步：按块表读回来。** `block_table.physical_slots()` 给出该请求**全部** token 的槽位（prefill 场景下正好就是刚写入的那 L 个），`gather` 读回连续的 `(L, H, D)`。

你可能会嘀咕：刚写进去又读回来，不是脱裤子放屁吗？——**在这个玩具实现里确实多此一举**，但请忍住别"优化"掉它。这个"写→读"的循环正是真实引擎的形状：真实 vLLM 的注意力 kernel 就是**只从仓库读**（历史是以前写的，不读仓库拿不到）。我们让 prefill 也走同一条路，是为了让两条路径共用一套存储逻辑，第 5 章的对拍才有意义。

**第 3 步：注意力。** 我们的 `scaled_dot_product_attention`（Week 2 写的）认的形状是 `(B, H, L, D)`，所以用 `permute(1, 0, 2)` 把 H 维换到前面、再 `unsqueeze(0)` 补一个大小为 1 的 batch 维——`permute` 是"按新顺序重排维度"，`unsqueeze` 是"插进一个长度为 1 的维度"。掩码用 Week 2 的因果掩码（不许偷看未来）。算完再把维度换回去、拼回头维度，`o_proj` 输出 `(L, hidden)`。

## 4.3 decode_batch：一口一口吃（批量版）

decode 阶段的引擎画面：一批请求**同时**各生成一个 token。难点在于它们历史长度不同——A 已有 10 个 token，B 只有 3 个。模型入口：

```python
@torch.no_grad()
def decode_batch(self, token_ids, positions, kv_cache, slot_mapping, block_tables):
    """一批请求各走一步。token_ids: (B,)，positions: (B,)。返回 (B, vocab)。"""
    x = self._embed(token_ids, positions)
    for block in self.blocks:
        x = block.decode_batch(x, kv_cache, slot_mapping, block_tables)
    return self._head(x)
```

- `token_ids` `(B,)`：每个请求**上一步生成的那个词**（这是本步的输入）；
- `positions` `(B,)`：每个请求这个新 token 的位置编号——注意**各请求的位置可以不一样**（A 在生成第 11 个词，B 在第 4 个），所以位置必须一个请求一个值；
- `block_tables`：B 个请求各自的 `BlockTable`（一个 list）。

注意力层是本章最精彩的部分：

```python
def decode_batch(self, x, kv_cache, slot_mapping, block_tables):
    """一批请求各生成一个 token。x: (B, hidden)。返回 (B, hidden)。"""
    B = x.shape[0]
    q = self.q_proj(x).view(B, self.num_heads, 1, self.head_dim)  # (B, H, 1, D)
    k_new = self.k_proj(x).view(B, self.num_heads, self.head_dim)
    v_new = self.v_proj(x).view(B, self.num_heads, self.head_dim)
    # 写入新 token 的 K/V（write 要 (n, H, D)）
    kv_cache.write(self.layer_idx, slot_mapping, k_new, v_new)

    slot_lists = [bt.physical_slots() for bt in block_tables]
    k_all, v_all, pad_mask = kv_cache.gather_padded(self.layer_idx, slot_lists)
    # k_all: (B, Lmax, H, D) -> (B, H, Lmax, D)
    k_all = k_all.permute(0, 2, 1, 3)
    v_all = v_all.permute(0, 2, 1, 3)
    mask = pad_mask.view(B, 1, 1, -1)  # 每个请求只看自己真实存在的词

    out, _ = scaled_dot_product_attention(q, k_all, v_all, mask)
    out = out.squeeze(2).reshape(B, -1)  # (B, H*D)
    return self.o_proj(out)
```

逐步看：

1. **q 的形状 `(B, H, 1, D)`**：每个请求本步只有**一个** query（刚生成的词回头看历史），序列维长度是 1；
2. **写入**：B 个新 token 的 K/V，形状 `(B, H, D)`，按 `slot_mapping`（B 个槽位号，每请求一个）写进各自的块——B 个请求的新数据**一笔写进同一个仓库**，互不干扰；
3. **读回**：`[bt.physical_slots() for bt in block_tables]` 收集 B 份槽位清单，交给上一章的 `gather_padded`——**短请求右补 0、拿到 mask**；
4. **掩码接力**：`pad_mask` 形状 `(B, Lmax)`，`view(B, 1, 1, -1)` 把它变成 `(B, 1, 1, Lmax)`。我们 Week 2 写的注意力函数里，掩码会和注意力分数做广播：每个请求的 query 只能看到 mask 为 True 的位置（自己的真实历史），补 0 的位置权重为 0。**因果掩码在这里不需要了**——query 是最新词，它的"过去"就是全部历史，本来就可以全看；
5. 输出 `squeeze(2)` 摘掉长度为 1 的序列维，拼回头维度，`o_proj` 得 `(B, hidden)`；模型入口再过 `_head` 得 `(B, vocab)`——每个请求一行下一个词的预测分数。

把 prefill 和 decode_batch 摆在一起对比，一张表看懂：

| | prefill | decode_batch |
|---|---|---|
| 服务对象 | 一个请求 | B 个请求 |
| 输入 | 整段提示词 `(L,)` | 每请求一个新词 `(B,)` |
| 写入槽位 | L 个 | B 个（每请求一个） |
| 读历史 | 自己刚写的 L 个槽位 | 各请求的全部槽位，补 0 对齐 |
| 掩码 | 因果掩码（不许看未来） | 补 0 掩码（不看假数据） |
| 返回 | `(vocab,)` | `(B, vocab)` |

> 💡 **你可能会问：decode_batch 每个请求只出一个 query，为什么不一个请求一个请求地算，非要拼成一批？**
>
> 因为 GPU 喜欢"大块头的活"。B 个请求拼成一批，矩阵乘法从 B 次小乘变成一次大乘，硬件利用率高得多——这正是 Week 5 continuous batching 提吞吐的核心机制。本周先把"一批能算对"这个地基打牢。

> ⚠️ **易踩坑：** 三个最常见的错位——(1) 忘了 `write` 的 k/v 要 `(n, H, D)` 而不是 `(B, H, 1, D)`；(2) `gather_padded` 返回的是 `(B, Lmax, H, D)`，进注意力前必须 `permute(0, 2, 1, 3)` 换成 `(B, H, Lmax, D)`；(3) mask 忘了 `view(B, 1, 1, -1)`，形状对不上广播不了。形状错了 PyTorch 会报 size mismatch，对着报错挨个 `print(shape)` 是最快的排查法。

> 📌 **对标 vLLM：** 真实 vLLM 的模型前向同样接收 `slot_mapping` 和块表信息（打包在 attention metadata 里），由 `vllm/v1/worker/` 里的执行层在每步组装好传给模型；prefill 和 decode 在真实 vLLM 里甚至可以混在一个 batch 里（chunked prefill，Week 8 会介绍）。分工边界和我们一致：**调度层管槽位，模型管计算**。

> 📌 **划重点：** 模型长出两个引擎接口：`prefill` 处理一个请求的整段提示词，`decode_batch` 让一批长短不一的请求共享同一个分页仓库各走一步。槽位从哪来、块表谁维护，都是调度层的事——模型只认 `slot_mapping` 和 `block_table` 这两张"仓库通行证"。

---

## 动手练习

1. 填出 `minivllm/model/transformer.py` 里的 6 个方法：`MultiHeadAttention.prefill` / `decode_batch`、`DecoderBlock.prefill` / `decode_batch`、`MiniTransformer.prefill` / `decode_batch`（层和模型两级的版本都是"薄包装"，照 4.2/4.3 的形状走）；
2. 跑：

```bash
.venv/bin/pytest tests/test_w4.py -k paged_equals_naive
```

   这条测试会用你的分页接口做贪心生成，再和 Week 3 的老路对比——**生成的词必须一模一样**。怎么个比法、为什么这么比，是下一章的全部内容。
3. （选做）在 `decode_batch` 里打一行 `print(k_all.shape, mask.shape)`，构造两个历史长度不同的请求跑一步，亲眼看看补 0 对齐后的形状。

## 参考答案

`reference/model/transformer.py`，搜索 `prefill` 和 `decode_batch`。**卡住 20 分钟再看**，重点对比注意力层里那几次 `view` / `permute` / `unsqueeze` 的顺序。

---

👉 下一章：[第 5 章：铁证——分页和不分页一字不差](./05-铁证-分页和不分页一字不差.md)
