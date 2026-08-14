# 第 2 章：KV cache——把中间结果存下来

> 本章你将：
> 1. 用"档案柜"的类比搞懂：注意力里到底**存什么**、为什么只存 K 和 V；
> 2. 逐行实现 `NaiveKVCache.append_and_get`——每层一对越攒越长的 K/V 张量；
> 3. 搞清楚 `torch.cat` 拼在哪个维度上，以及"拼接"暗藏的代价（Week 4 的伏笔）；
> 4. 跑绿 `pytest tests/test_w3.py -k naive_cache`。

---

## 2.1 先想清楚：要存什么？

上一章的结论：老词重算出来的东西和上次一模一样，所以"算过的存下来"就行。
但一次前向算出的中间结果多着呢——嵌入、每层的隐藏状态、Q、K、V、注意力分数……
全存吗？存哪个才够用？

回到 Week 2 的注意力。生成下一个词时，本质上是**最新那个词**拿着自己的 Q 去问：
"前面每个词，你们各自是什么（K）、能提供什么内容（V）？"然后按匹配度加权汇总：

$$
\text{注意力输出} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

注意这个式子的"不对称"：

- **Q 只需要最新的一个。** 老词的 Q 是用来"它当时提问"的，现在它不提新问题了，
  它的 Q 再也没用了；
- **K 和 V 需要全部的。** 因为新词要回头看**每一个**老词，就得有每个老词的
  标签（K）和内容（V）。

所以答案非常精确：**每层、每个词，存一份 K 和一份 V。Q 不存，用完即弃。**

> 📌 **划重点：** KV cache 存的是"每个词在每一层的 K 和 V"。这就是为什么它叫
> KV cache，而不是 Q cache、QKV cache。

## 2.2 类比：每个词的档案

把每一层的注意力想象成一个**资料室**：

- 每个词第一次经过这层时，办事员给它建一份**档案**：封面标签是 K（"我是谁、
  我管什么事"），袋里的材料是 V（"找我你能拿到什么"）；
- 之后任何新词来查资料，办事员**不再重新政审老词**，直接翻出它们的档案；
- 新词自己也要建档（它的 K/V 存进柜子），供将来的词查。

cache 就是那一排档案柜。`NaiveKVCache` 的"naive"体现在：柜子是一根**连续的长货架**，
每来一份新档案，就把旧档案全搬到一个更长的货架上（`torch.cat` 拼接）——
能用，但搬来搬去很笨。先忍它一周，Week 4 我们换"分页储物柜"。

## 2.3 NaiveKVCache：每层一对 K/V

打开 `minivllm/cache/kv_cache.py`，骨架已经给你了：

```python
class NaiveKVCache:
    """每层一对 (K, V)，形状 (batch, heads, seq_len, head_dim)，逐 token 变长。"""

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.k_list = [None] * num_layers  # 每层一个 (B, H, S, D) 或 None
        self.v_list = [None] * num_layers
```

结构一目了然：**每一层一个档案柜**，`k_list[i]` 是第 `i` 层的所有 K，
形状 `(B, H, S, D)`——batch、头数、序列长度、每头维度。刚建好时全是 `None`
（一个词都还没见过）。

三个方法里，`seq_len` 和 `reset` 已经帮你写好了：

```python
def seq_len(self, layer_idx=0):
    """该层已经缓存了多少个 token。"""
    k = self.k_list[layer_idx]
    return 0 if k is None else k.shape[2]   # S 在第 2 维

def reset(self):
    """清空全部缓存。"""
    self.k_list = [None] * self.num_layers
    self.v_list = [None] * self.num_layers
```

你的任务是核心的那个：`append_and_get`。

## 2.4 实现 append_and_get

这个方法的合同（docstring 里写着的）：

```python
def append_and_get(self, layer_idx, k, v):
    """把该层新算的 (k, v) 拼到历史后面，存起来，返回完整 (k, v)。

    k, v: (B, H, L_new, D)。返回 (B, H, L_old + L_new, D)。
    """
```

拆成三步，一步都不能少：

1. **取出旧档案**：`old_k = self.k_list[layer_idx]`；
2. **如果有旧的，把新的拼在后面**：`torch.cat([old_k, k], dim=2)`。
   `dim=2` 是序列维 $S$——新词当然排在老词**后面**。V 同理；
3. **存回柜子，并返回完整的 K/V**——注意返回的是"旧 + 新"的**完整**版本，
   因为注意力要用全部历史来算。

参考答案的全部逻辑就这几行（先自己写！）：

```python
old_k = self.k_list[layer_idx]
if old_k is not None:
    k = torch.cat([old_k, k], dim=2)
    v = torch.cat([self.v_list[layer_idx], v], dim=2)
self.k_list[layer_idx] = k
self.v_list[layer_idx] = v
return k, v
```

> ⚠️ **易踩坑：** 三个高频错误——
> ① `dim` 写错：K 的形状是 `(B, H, S, D)`，拼的是第 2 维（`dim=2`），
> 不是 `dim=1`（那是头维，拼上去头数就变了）；
> ② 忘了**存回** `self.k_list[layer_idx]`，下次来还是空的；
> ③ 只返回新算的 `k` 而不是拼接后的完整版——那注意力就只看得见新词一个词了。

## 2.5 用测试验证你的实现

`tests/test_w3.py` 里的 `test_naive_cache_append_and_len` 把上面的合同钉死了，
我们逐段读一遍（这也是你调试时的对照表）：

```python
cache = kv_mod.NaiveKVCache(num_layers=2)
assert cache.seq_len() == 0                    # 刚建好，空空如也

k1 = torch.ones(1, 2, 3, 4)                    # (B=1, H=2, S=3, D=4)：先来 3 个词
v1 = torch.zeros(1, 2, 3, 4)
k, v = cache.append_and_get(0, k1, v1)
assert k.shape == (1, 2, 3, 4)                 # 第一次没有旧档案，返回的就是它自己
assert cache.seq_len() == 3

k2 = torch.ones(1, 2, 1, 4) * 2                # 又来 1 个新词（S=1）
k, v = cache.append_and_get(0, k2, k2)
assert k.shape == (1, 2, 4, 4)                 # 3 + 1 = 4，拼上了
assert cache.seq_len() == 4
assert torch.equal(k[0, 0, :3], torch.ones(3, 4))   # 旧的在前
assert torch.equal(k[0, 0, 3], torch.full((4,), 2.0))  # 新的在后

assert cache.seq_len(1) == 0                   # 第 1 层没人动过，还是空的
cache.reset()
assert cache.seq_len() == 0                    # 清空归零
```

两个细节值得停下来看一眼：

- **"旧的在前、新的在后"**被显式断言了。顺序很重要：注意力分数矩阵的第 $i$
  列对应第 $i$ 个词，位置排错了，词和词就张冠李戴了；
- **各层独立**。第 0 层存了 4 个词，第 1 层还是 0——`layer_idx` 参数就是用来
  区分"这是哪一层的档案柜"的。

> 💡 **你可能会问：为什么方法名叫 append_and_get，append 完直接 return 不就行了，"get"在哪？**
>
> "get"就体现在返回值是**完整历史**而不是新增部分。调用方（注意力层）拿到返回值后
> 直接拿去算注意力，不需要再回头访问 cache。一个方法把"存"和"取完整版"合并了，
> 调用处就只用写一行。

## 2.6 一句话交代：cat 的代价

`torch.cat` 每次都要**新申请一块更大的内存，把旧数据整个拷过去**。序列 1000 个词时，
第 1001 个词到来，要把前 1000 个词的 K/V 全搬一次家；第 1002 个词再搬一次……
搬家总开销又是平方级的。

本周我们忍它——因为比起"重算整段前向"，搬家的开销小得多，3 倍加速照样到手
（第 4 章实测）。但请记住这根刺：**连续存放 + 反复搬运**，这正是 Week 4
"分页 KV cache"要拔掉的刺。

> 📌 **对标 vLLM：** 真实 vLLM 的 KV cache 不用 `cat` 搬家，而是预分配一大块显存、
> 按固定大小的"块"（block）写入，逻辑位置到物理块的映射记在块表里——对应
> `vllm/v1/core/kv_cache_manager.py` 和 `vllm/v1/core/block_pool.py`。
> 那就是 Week 4 的主角，本周先记住"naive 的思想是对的，实现是笨的"。

> 📌 **划重点：** KV cache = 每层一对越攒越长的 K/V 张量；`append_and_get` 负责
> "拼进去、存起来、返回完整版"。Q 不存，因为只有最新的词要提问。

---

## 动手练习

1. 在 `minivllm/cache/kv_cache.py` 里实现 `NaiveKVCache.append_and_get`
   （照 2.4 节的三步，先别看答案）；
2. 跑：

```bash
.venv/bin/pytest tests/test_w3.py -k naive_cache
```

3. （思考题，不写代码）如果序列有 1000 个词、`num_layers=2`、每层的 K 形状是
   `(1, 4, 1000, 16)`，这个 cache 一共存了多少个浮点数？提示：K 和 V 各一份，
   每层一份。

## 参考答案

`reference/cache/kv_cache.py` 里的 `NaiveKVCache` 是标准答案，整个类只有 30 行。
**卡住 20 分钟再看**，重点对照：你拼的维度是不是 `dim=2`，存回和返回的是不是
"拼接后的完整版"。

---

👉 下一章：[第 3 章：prefill 与 decode——一次吃饱与一口一口吃](./03-prefill与decode-一次吃饱与一口一口吃.md)
