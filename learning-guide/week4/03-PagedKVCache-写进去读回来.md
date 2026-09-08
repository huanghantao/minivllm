# 第 3 章：PagedKVCache——写进去，读回来

> 本章你将：
> 1. 认识所有请求共享的分页仓库：`k_cache` / `v_cache` 的形状为什么是 `(num_blocks, block_size, num_heads, head_dim)`；
> 2. 亲手实现 `write` 和 `gather`——上一章的物理槽位公式，在这里变成一行张量索引；
> 3. 读懂已给好的 `gather_padded`：一批长短不一的请求，怎么补 0 对齐、怎么用掩码装傻；
> 4. 跑通 `pytest tests/test_w4.py -k paged`。

---

## 3.1 仓库的结构：每层一对"柜子墙"

上一章的 `BlockPool` 只管编号，真正的 K/V 数据住在 `PagedKVCache` 里。它是**全局唯一**的仓库：不管同时跑几个请求，所有请求、所有层的 K/V 都存在这里，靠槽位号各找各的。

`__init__` 已经给好，核心就两行：

```python
shape = (num_blocks, block_size, num_heads, head_dim)
self.k_cache = [
    torch.zeros(shape, device=device, dtype=dtype)
    for _ in range(num_layers)
]
self.v_cache = [
    torch.zeros(shape, device=device, dtype=dtype)
    for _ in range(num_layers)
]
```

读一下这个设计：

- `k_cache` 是一个 **list，一层一个张量**（`v_cache` 同理）。每层一对，和 Week 3 的 `NaiveKVCache` 一样；
- 每个张量的形状是 `(num_blocks, block_size, num_heads, head_dim)`——四个维度念出来就是："**第几块**里的、**块内第几格**的、**第几个注意力头**的、**head_dim 个数**"。前两维就是储物柜的"第几号柜、柜里第几格"；
- 开局全部 `torch.zeros`：先把柜子墙砌好，格子空着。

仓库一共占多少内存？`2 × num_layers × num_blocks × block_size × num_heads × head_dim × 每个数的字节数`——**开多大就是多大，和有多少请求在跑无关**。这正是"分页"和"按需分配"的差别：柜子墙一次砌好，但空格随领随用，谁走谁还。

## 3.2 公式落地：view 一下，槽位就是下标

上一章我们说：物理槽位 = 块号 × 块大小 + 块内偏移。现在看它怎么变成代码。

张量有个视图操作 `view`：**不改任何数据，只改变"看法"**（Week 1 见过）。对 `(num_blocks, block_size, H, D)` 的张量做 `view(-1, H, D)`（`-1` 表示"这一维多大自己算"），前两维就被压平成一维：

```python
flat_k = self.k_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
# 形状：(num_blocks * block_size, H, D)
```

压平后的第 `s` 行是谁？正是"块号 × 块大小 + 偏移 = s"那一格——因为张量在内存里本来就是按顺序排的，第 0 块的 4 格是第 0~3 行，第 1 块的 4 格是第 4~7 行……**上一章的公式不是巧合，它就是 `view` 压平后的行号。** 所以物理槽位 `s` 直接当行下标用：`flat_k[s]`。

## 3.3 你要实现的两个方法

### `write`：按槽位写进去

模型每算出一批 K/V（prefill 时一批、decode 时一个），调用方会同时给一张 `slot_mapping`："第 0 个 token 写进槽位 12、第 1 个写进槽位 13……"。`write` 就照着写：

```python
def write(self, layer_idx, slot_mapping, k, v):
    """把 n 个 token 的 K/V 写进指定槽位。

    slot_mapping: (n,) 的 LongTensor，物理槽位号；
    k, v: (n, num_heads, head_dim)。
    """
    slots = torch.as_tensor(slot_mapping, dtype=torch.long, device=k.device)
    flat_k = self.k_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
    flat_v = self.v_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
    flat_k[slots] = k.to(self.dtype)
    flat_v[slots] = v.to(self.dtype)
```

三个要点：

1. `torch.as_tensor(slot_mapping, dtype=torch.long, ...)`：调用方可能传 list 也可能传 tensor，先统一成 **LongTensor**。PyTorch 里"用一批下标取值/赋值"要求下标必须是整数（long）类型；
2. `flat_k[slots] = k`：这是 PyTorch 的**高级索引**——左边给一个下标列表，右边给同样行数的数据，逐行对位写进去。`flat_k[[5, 2, 9]] = k` 的意思是"k 的第 0 行写进第 5 行、第 1 行写进第 2 行、第 2 行写进第 9 行"。**顺序完全由 slots 决定，爱写哪写哪**——乱序也没事；
3. `.to(self.dtype)`：模型算的 dtype（比如 float32）和仓库的 dtype（可能将来是 float16）不一致时，顺手转一下，防止类型不匹配报错。

### `gather`：按槽位读回来

读取就是写入的镜像：

```python
def gather(self, layer_idx, slots):
    """按槽位号把 K/V 读回来，拼成连续张量。

    slots: list[int]，长度 L；
    返回 (k, v)，形状都是 (L, num_heads, head_dim)。
    """
    idx = torch.as_tensor(slots, dtype=torch.long, device=self.device)
    flat_k = self.k_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
    flat_v = self.v_cache[layer_idx].view(-1, self.num_heads, self.head_dim)
    return flat_k[idx], flat_v[idx]
```

`flat_k[idx]` 还是高级索引：`idx = [12, 13, 28]` 就返回这三行拼成的新张量，**顺序按 idx 来**。所以调用方只要把 `BlockTable.physical_slots()`（逻辑顺序的槽位清单）传进来，拿到的就是**逻辑顺序的连续 K/V**——散着存、连着读，上一章承诺的"鱼与熊掌"就在这里兑现。

测试 `test_paged_cache_write_gather_roundtrip` 把这个性质钉得很死：故意用乱序槽位 `[5, 2, 9]` 写入，再按 `[5, 2, 9]` 读回来必须**一字不差**；再换个顺序 `[9, 5]` 去读，回来的就该是换过顺序的那两行。

## 3.4 已给好的 `gather_padded`：长短不一，补 0 对齐

单请求的读取搞定了，但引擎里 decode 是**一批请求一起走**（第 4 章的 `decode_batch`）。麻烦在于：这批请求历史长度各不相同——A 已经 10 个 token，B 才 3 个。PyTorch 的批量计算要求大家形状一致，怎么办？

**补齐（padding）**：短的请求右侧补 0，对齐到最长的那个；再给一张**掩码**（mask）记下"哪些位置是真数据、哪些是补的"，算注意力时让模型对补的位置视而不见（Week 2 的因果掩码是同一个语义：True = 允许看）。

`gather_padded` 已经给好了，读懂它就行：

```python
def gather_padded(self, layer_idx, slot_lists):
    """slot_lists: B 个请求各自的物理槽位序列。
    返回 (k, v, mask)：k, v 形状 (B, max_len, H, D)；
    mask 形状 (B, max_len)，True 表示该位置是真实数据。"""
    B = len(slot_lists)
    lens = [len(s) for s in slot_lists]
    max_len = max(lens)
    k = torch.zeros(
        B, max_len, self.num_heads, self.head_dim,
        device=self.device, dtype=self.dtype,
    )
    v = torch.zeros_like(k)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=self.device)
    for b, slots in enumerate(slot_lists):
        kk, vv = self.gather(layer_idx, slots)
        k[b, : len(slots)] = kk
        v[b, : len(slots)] = vv
        mask[b, : len(slots)] = True
    return k, v, mask
```

逻辑一目了然：

1. 先找到最长长度 `max_len`，造三张全 0 的张量：`k`、`v`（`(B, max_len, H, D)`）和布尔型的 `mask`（`(B, max_len)`，全 False）；
2. 逐个请求 `gather` 出它的真实 K/V，**贴到左边**，右边的剩余位置保持 0；
3. `mask[b, :len(slots)] = True`：真实数据的位置打 True，补 0 的位置保持 False。

注意这个函数是**建筑在你写的 `gather` 之上**的——你的 `gather` 不对，它也跟着错。这也是为什么测试顺序是先测 `write`/`gather`，再测 `gather_padded`。

看一眼测试 `test_gather_padded` 里的预期，体会"左对齐、右补 0"：往槽位 0~3 写了全 1 的 K，然后 `gather_padded(0, [[0, 1, 2], [3]])`——两个请求，一个 3 个 token、一个 1 个 token。结果：

- `k.shape == (2, 3, 4, 8)`（补齐到 max_len=3）；
- `mask.tolist() == [[True, True, True], [True, False, False]]`；
- 第二个请求的 `k[1, 0]` 是全 1（真数据），`k[1, 1]` 是全 0（补的）。

> 💡 **你可能会问：补上去的 0 会不会污染注意力计算？**
>
> 不会。Week 2 讲过掩码的语义：被掩掉的位置，注意力权重会被强行压成 0（softmax 之前填负无穷），那些位置的 V 是什么值根本不重要——0 也好、随机数也好，反正乘上权重 0。下一章 `decode_batch` 里你会亲眼看到这张 mask 怎么接进注意力。

> ⚠️ **易踩坑：** `view(-1, H, D)` 只是"换个看法"，它和原张量**共享同一块内存**。这恰恰是我们要的——往 `flat_k` 里写，就是往 `k_cache[layer_idx]` 里写。但反过来也意味着：别指望 `view` 帮你复制数据，它不是副本。

> 📌 **对标 vLLM：** 真实 vLLM 的 KV cache 张量形状和我们几乎一样（多了些对 GQA、kv 合并的考虑）。写入动作在真实 vLLM 里是一个 CUDA kernel：`reshape_and_cache`，可以在 `vllm/v1/attention/backends/flash_attn.py` 里看到它的调用——模型每步算完 K/V，kernel 按 `slot_mapping` 把数据原地写进分页仓库，和我们的 `write` 一模一样，只是用 CUDA 写、快得多。

> 📌 **划重点：** 分页仓库 = 每层一对 `(num_blocks, block_size, H, D)` 的张量；`view(-1, H, D)` 一压平，物理槽位直接变成行下标；`write` 用高级索引按槽位写、`gather` 按槽位读，`gather_padded` 再把一批请求补齐对齐。散着存、连着读，存储自由和计算方便两头都占。

---

## 动手练习

1. 填出 `minivllm/cache/paged_cache.py` 的 `PagedKVCache.write` 和 `PagedKVCache.gather`；
2. 跑：

```bash
.venv/bin/pytest tests/test_w4.py -k paged
```

   `test_paged_cache_write_gather_roundtrip`、`test_paged_cache_layers_are_independent`、`test_gather_padded` 三条应该全绿。（如果 `-k paged` 顺带带红了 `test_paged_equals_naive`，别慌——那是对拍测试，需要第 4 章的模型接口，下一章再收拾它。）
3. （选做）写一个三行小实验验证"层与层互不相干"：往第 0 层写数据，从第 1 层同样的槽位读，读出来的应该是全 0。想想为什么这个性质很重要（提示：如果两层共享了内存，会发生什么？）

## 参考答案

`reference/cache/paged_cache.py`。**卡住 20 分钟再看**，重点对比你的高级索引写法：左边是"下标列表"，右边是"同样行数的数据"，一一对位。

---

👉 下一章：[第 4 章：模型的分页接口——prefill 与 decode_batch](./04-模型的分页接口-prefill与decode_batch.md)
