# 第 5 章：RoPE 与 GQA——位置编码与省内存

> 本章你将：
> 1. 用最朴素的方式理解 **RoPE（旋转位置编码）**：把向量看成一堆小指针，
>    位置越靠后转得越多；
> 2. 实现 `rotate_half`、`apply_rotary_pos_emb`、`RotaryEmbedding.forward`；
> 3. 搞懂 **GQA（分组查询注意力）**：K/V 头减半，KV cache 直接省一半，
>    以及补齐头数的那一行 `repeat_interleave`；
> 4. 一句话认识 Qwen3 的小补丁 **QK-Norm**。

---

## 5.1 位置编码的老办法和它的天花板

先回忆 Week 2：注意力本身是"脸盲"的——它只看词的内容，不知道谁前谁后。
所以 MiniTransformer 给每个位置编号准备了一行**可学习的位置嵌入**，
加到词向量上：位置 0 查表加一行，位置 1 加另一行……

这个"查表"法有个明显的天花板：

- 表是**背下来**的。训练时最多见过 128 个位置，第 129 个位置的向量
  模型压根没学过——**外推能力差**；
- 它编码的是**绝对位置**（"你是第 5 个"），但语言里重要的是**相对距离**
  （"它"在前面 3 个词之内，大概率能找到指代对象）。

RoPE（Rotary Position Embedding，旋转位置编码）用一个优雅的几何动作
同时治好这两个病。它不给词向量**加**什么，而是把它**转**一下。

## 5.2 RoPE 最朴素的讲法：一排转速不同的指针

只看核心直觉，不做任何推导。

**第一步：两两配对。** Qwen3 的 `head_dim` 是 128，把每个向量的 128 个数
配成 64 对：第 1、2 个数一对，第 3、4 个一对……每一对看成平面上的
一根**指针**（一个二维小向量）。

**第二步：按位置旋转。** 一个词在序列里是第 $pos$ 个，就把它身上第 $i$ 对
指针转 $pos \times \theta_i$ 弧度。转多少**正比于位置编号**——位置越靠后，
转得越多。

**第三步：每对转速不同。** 64 对指针的转速 $\theta_i$ 按几何级数排开：

$$
\theta_i = \text{base}^{-2i / d}, \quad \text{base} = 1{,}000{,}000
$$

靠前的对转得飞快，靠后的对转得极慢——想象钟表：秒针、分针、时针
转速各不相同，三根针的朝向合起来唯一确定一个时刻。64 根指针的朝向
合起来，就是这个词的"**位置指纹**"。

为什么这样就够了？两个直觉（记住结论即可）：

1. **旋转不改变长度**，只改朝向——词的内容信息（模长）不被破坏；
2. 两根指针的**夹角只取决于位置之差**（第 5 个词和第 8 个词，无论从哪
   开始数，指针朝向差都一样）——模型天然学到的是**相对距离**。
   所以序列变长时它也不慌：第 1000 个词无非是多转几圈的事，
   不需要"背过"第 1000 行。

## 5.3 代码里的旋转：rotate_half + apply_rotary_pos_emb

旋转的数学是二维旋转公式（不推导，认个脸熟）：向量 $(x, y)$ 转 $\theta$
弧度后变成 $(x\cos\theta - y\sin\theta,\ x\sin\theta + y\cos\theta)$。

代码里把 64 对打包批量处理。`rotate_half` 负责准备"配对交换"的那一半：

```python
def rotate_half(x):
    """把最后维对半切开、交换并取负：[x1, x2] -> [-x2, x1]。"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
```

128 维切成两半：前半 $x_1$、后半 $x_2$（各 64 维，一一配对），
返回拼起来的 $[-x_2, x_1]$。然后一行施加旋转：

```python
def apply_rotary_pos_emb(t, cos, sin):
    """给 q/k 施加旋转。t: (..., head_dim)，cos/sin 可广播到 t。"""
    return t * cos + rotate_half(t) * sin
```

对照一下：`t * cos + rotate_half(t) * sin` 展开正是
$(x_1\cos\theta - x_2\sin\theta,\ x_2\cos\theta + x_1\sin\theta)$——
64 对二维旋转一次写完。（配对方式是"前半配后半"而不是"相邻配对"，
所以 `RotaryEmbedding` 里要把 freqs 自己跟自己 `cat` 一份，马上看到。）

cos/sin 从哪来？`RotaryEmbedding` 预存转速、现用现算：

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        """positions: (n,) 的整数位置。返回 (cos, sin)，形状 (n, head_dim)。"""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # (n, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (n, D)，rotate_half 的配对方式
        return emb.cos(), emb.sin()
```

你要实现的是 `forward`，三步：

1. **`positions[:, None] * inv_freq[None, :]`**：外积——$n$ 个位置
   × 64 种转速，得到每个位置每对指针该转的角度，形状 `(n, 64)`；
2. **`torch.cat([freqs, freqs], dim=-1)`**：复制一份拼成 `(n, 128)`——
   因为配对是"前半配后半"，前后两半用同一组角度；
3. **取 cos 和 sin 返回**。

`register_buffer` 顺带一句：它把 `inv_freq` 登记为"模型的一部分但不是
参数"——跟着模型搬家（`.to(device)`），但不参与训练。

> ⚠️ **易踩坑：** RoPE 只施加在 **q 和 k** 上，**不施加在 v 上**。
> 直觉：注意力靠 q·k 的点积算匹配度，位置信息需要混进"匹配"这一步；
> v 是被加权汇总的内容，带上旋转反而污染内容。写 `naive_forward` 时
> 别把 v 也转了。

## 5.4 GQA：K/V 头减半，cache 省一半

### 从 MHA 到 GQA

Week 2 的多头注意力（MHA）里，Q、K、V 头数一样多：每个 Q 头配自己
专属的 K/V 头。Week 3 算过账：KV cache 的大小正比于 **K/V 的头数**。

GQA（Grouped-Query Attention，分组查询注意力）的想法简单粗暴：

> 让几个 Q 头**合租**一对 K/V 头。

Qwen3-0.6B：16 个 Q 头，8 对 K/V 头——每 2 个 Q 头合租一对。
`num_kv_groups = 16 / 8 = 2`。还记得 Week 3 算的"每 token 112 KB"吗？
那已经是 GQA 打过 5 折的价格；如果用 MHA（16 对 KV 头），就是 224 KB。
更激进的 MQA 只留 1 对 KV 头，能打 1/16 折。

效果损失呢？实践结论：略微掉点，但换来 cache 减半、解码更快，
**性价比极高**——LLaMA、Qwen 全线采用。

### 实现：一行 repeat_interleave

算注意力时，头数必须对齐（16 个 Q 头各要一份 K）。所以把 8 对 K/V
**复制**成 16 份再算——复制没有信息增量，只是为了让形状配得上：

```python
def _repeat_kv(self, t):
    """GQA 的关键：把 K/V 头复制 num_kv_groups 份，凑成和 Q 一样多。

    t: (n, Hkv, D) -> (n, H, D)
    """
    return t.repeat_interleave(self.num_kv_groups, dim=1)
```

`repeat_interleave`（新操作一句话）：沿指定维度把每个元素**连续复制** n 份。
`[a, b]` 复制 2 份是 `[a, a, b, b]`——第 0、1 个 Q 头合租第 0 对 K/V，
第 2、3 个 Q 头合租第 1 对……

> ⚠️ **易踩坑：** 别用 `t.repeat(1, 2, 1)` 代替——`repeat` 是**整体平铺**，
> 得到 `[a, b, a, b]`，配对关系全错（第 1 个 Q 头会配上第 0 对 KV 的
> 复制体，第 2 个配上第 1 对……错位了）。记口诀：
> `repeat_interleave` = `[a,a,b,b]`（合租），`repeat` = `[a,b,a,b]`（平铺）。

> 💡 **你可能会问：复制完再算注意力，那 cache 里存的是复制前还是复制后？**
>
> 存的是**复制前**（8 头）。cache 在 `kv_cache.write` 里存的是
> `_project_qkv` 刚投影出来的 K/V，`_repeat_kv` 是在**读出来之后、
> 算注意力之前**临时展开的。省内存省在存储环节，计算环节的复制是
> 免费的广播逻辑（真实 kernel 连复制都省了，直接让两个 Q 头去读
> 同一份 K/V——又是"思想对、实现可以更精"的例子）。

## 5.5 一句话补丁：QK-Norm

Qwen3 还有个小补丁：q、k 投影完之后、旋转之前，各过一个小小的
RMSNorm（在 `head_dim=128` 这一维上）：

```python
self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
```

作用一句话：**稳住 q/k 的数值大小，注意力分数不会随训练越滚越大。**
没有推导，把它当成"出厂自带的稳定器"即可。但要记得它的存在——
第 6 章加载权重时，`q_norm.weight` / `k_norm.weight` 是 checkpoint 里
实打实的两个张量，漏了它们对拍就对不上。

> 📌 **对标 vLLM：** RoPE 在 `vllm/model_executor/layers/rotary_embedding.py`
> （含各种变体），GQA 的展开逻辑藏在注意力 kernel 里
> （`vllm/attention/layer.py`，直接用分组索引读 K/V，不做物理复制），
> Qwen3 的 QK-Norm 在 `vllm/model_executor/models/qwen3.py` 的
> `Qwen3Attention.__init__` 里——和我们的代码逐行对应。

> 📌 **划重点：** RoPE = 把向量两两配成 64 对小指针，按位置编号旋转，
> 转速按几何级数排开，只转 q/k；GQA = 几个 Q 头合租一对 K/V，
> cache 头数减半，算注意力前用 `repeat_interleave` 补齐。

---

## 动手练习

1. 在 `minivllm/model/qwen3.py` 里实现 `rotate_half`、
   `apply_rotary_pos_emb`、`RotaryEmbedding.forward` 和
   `Qwen3Attention._repeat_kv`；
2. 写个 30 秒的自检脚本（终极验收仍是第 6 章的对拍）：

```bash
.venv/bin/python - <<'EOF'
import torch
from minivllm.model.qwen3 import RotaryEmbedding, apply_rotary_pos_emb

rope = RotaryEmbedding(head_dim=8, base=10000)
cos, sin = rope(torch.arange(5))
print("cos 形状：", cos.shape)          # (5, 8)

# 旋转不改变向量长度（RoPE 的核心性质）
x = torch.randn(5, 8)
y = apply_rotary_pos_emb(x, cos, sin)
print("旋转前后长度：")
print(x.norm(dim=-1))
print(y.norm(dim=-1))                    # 两行应该几乎一样

# 位置 0 不转（角度为 0：cos=1, sin=0）
print("位置 0 转完等于自己：",
      torch.allclose(y[0], x[0], atol=1e-6))
EOF
```

3. （手算题）Qwen3-0.6B 若改回 MHA（16 对 KV 头），每 token 的 KV cache
   是多少 KB？用 Week 3 的公式：$28 \times 2 \times H \times 128 \times 2$ 字节。

## 参考答案

`reference/model/qwen3.py` 里的 `RotaryEmbedding`、`rotate_half`、
`apply_rotary_pos_emb`、`_repeat_kv` 是标准答案，加起来不到 15 行。
**卡住 20 分钟再看**，重点对照：`freqs` 有没有 `cat` 两份、
`repeat_interleave` 的 `dim` 是不是 1。

---

👉 下一章：[第 6 章：加载真实权重——和 HuggingFace 对拍](./06-加载真实权重-和HuggingFace对拍.md)
