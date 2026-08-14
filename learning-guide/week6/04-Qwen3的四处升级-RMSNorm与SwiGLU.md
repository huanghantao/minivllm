# 第 4 章：Qwen3 的四处升级——RMSNorm 与 SwiGLU

> 本章你将：
> 1. 拿到一张"MiniTransformer → Qwen3 升级清单"（四处，全局地图）；
> 2. 搞懂第一处升级：**RMSNorm**——LayerNorm 的"减配版"，为什么减了配反而更好；
> 3. 搞懂第四处升级：**SwiGLU**——MLP 从"两个矩阵"变"三个矩阵 + 一道门"；
> 4. 亲手实现 `RMSNorm.forward` 和 `Qwen3MLP.forward`。

---

## 4.1 先看全局地图：只差四处升级

Week 2 你手写的 MiniTransformer 已经包含 Transformer 的全部招式：
嵌入、注意力、归一化、MLP、残差、输出头。真实模型 Qwen3 并没有发明
新招式，只是把其中四个零件各换了一个改良版：

| # | 零件 | MiniTransformer（教学版） | Qwen3（真实版） | 为什么换 |
|---|---|---|---|---|
| 1 | 归一化 | LayerNorm | **RMSNorm** | 少算一步，更快，效果相当（本章） |
| 2 | 位置编码 | 学习的位置嵌入（查表） | **RoPE** 旋转位置编码 | 相对位置、外推性好（下一章） |
| 3 | 注意力 | MHA（Q/K/V 头一样多） | **GQA**（K/V 头减半） | KV cache 直接减半（下一章） |
| 4 | MLP | 升维 + GELU + 降维 | **SwiGLU**（三路门控） | 同样参数表达力更强（本章） |

本章解决 1 和 4——它们只涉及单个模块内部，最好上手；下一章解决 2 和 3。
到第 6 章，拼起来的完整模型要加载真实权重、和 HuggingFace 对拍。

先看 Qwen3-0.6B 的真实身材（它的 `config.json`，第 7 章逐字段讲）：

| 配置项 | 值 |
|---|---|
| 隐藏维度 `hidden_size` | 1024 |
| 层数 `num_hidden_layers` | 28 |
| 注意力头数 `num_attention_heads` | 16 |
| KV 头数 `num_key_value_heads` | 8 |
| 每头维度 `head_dim` | 128 |
| MLP 中间维 `intermediate_size` | 3072 |
| 词表 `vocab_size` | 151 936 |

## 4.2 升级一：RMSNorm——不减均值的归一化

### 回忆 LayerNorm 在干什么

Week 2 讲过：归一化是"把每行的数值重新拉回标准身材"，防止数字在几十层
传递中越滚越大或越缩越小。LayerNorm 做两件事：

1. 减均值（把这一行的中心挪到 0）；
2. 除标准差（把这一行的胖瘦压成 1）。

（外加一组可学习的缩放 weight 和偏移 bias。）

### RMSNorm 的发现：第一步可以省掉

研究者们后来发现：**"减均值"这一步对效果几乎没贡献，删掉也不掉点。**
只留下"把数值大小压回正常范围"这一步。怎么衡量"大小"？用均方根
（Root Mean Square，RMS）——平方、平均、开根号：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot w
$$

逐符号读：

- $x^2$：每个元素平方（负数也变正，只反映"离 0 多远"）；
- $\text{mean}(x^2)$：这一行平方的平均——"这一行平均有多胖"；
- 开根号除回去：把整行缩放到"平均胖瘦约为 1"；
- $\epsilon$（`rms_norm_eps`，Qwen3 是 $10^{-6}$）：防止分母为 0 的保险丝；
- $w$：可学习的缩放（每维一个数，和 LayerNorm 的 weight 一样）。

对比表：

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | ✔ | ✘ |
| 除尺度 | 标准差 | 均方根 |
| 可学习参数 | weight + bias | 只有 weight |
| 代价 | 多一次减法、多一组参数 | 更省 |

省得不多，但归一化在模型里出现次数极多（Qwen3-0.6B 有
$28 \times 2 + 1 = 57$ 处，还不算下章的 QK-Norm），聚沙成塔。

### 实现 RMSNorm.forward

骨架在 `minivllm/model/qwen3.py`，`__init__` 已给出（就一个 weight 参数，
初始化为全 1）。你要写的是 `forward`：

```python
def forward(self, x):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x * self.weight.float()).to(dtype)
```

逐行看（一个新 torch 操作）：

1. **先记下原 dtype，转 float32**：真实权重是 bf16（半精度），半精度下
   平方再求和容易丢精度甚至溢出。归一化这种"先平方再开根"的操作对精度
   敏感，所以先升级到 float32 算完再转回去——这是工业实现的标配动作；
2. **`x.pow(2).mean(dim=-1, keepdim=True)`**：沿最后一维（特征维）算
   平方的均值。`keepdim=True` 保持维度不塌，方便下一步广播；
3. **`torch.rsqrt`**：一句话——"1 除以开根号"（reciprocal sqrt），
   即 $\frac{1}{\sqrt{\cdot}}$。乘上它等于除公式里那个分母；
4. **乘 weight、转回原 dtype**：可学习缩放在最后乘上。

> ⚠️ **易踩坑：** ① `mean` 的维度写错：归一化沿**最后一维**（特征维）
> 做，不是序列维；② 忘了 `+ self.eps`，输入恰好全 0 时除零；
> ③ 忘转回 `.to(dtype)`，输出的 dtype 和权重对不上，
> 下一层矩阵乘直接报类型错误。

## 4.3 升级四：SwiGLU——给 MLP 装一道门

### MiniTransformer 的 MLP 长什么样

Week 2 的 MLP 是两个矩阵：先把 1024 维升到 4 倍宽，过 GELU 激活，
再压回 1024 维：

$$
\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x)
$$

直觉：升到高维空间"展开思考"，激活函数提供非线性，再压回来。

### SwiGLU：三路矩阵 + 一道乘法门

Qwen3 的 MLP 有**三个**矩阵，公式是：

$$
\text{SwiGLU}(x) = \text{down}\Big(\text{silu}(\text{gate}(x)) \odot \text{up}(x)\Big)
$$

把三个矩阵拟人化一下：

- **`up_proj`**（$1024 \to 3072$）：像老 MLP 的 $W_1$，负责"展开内容"；
- **`gate_proj`**（$1024 \to 3072$）：和 up 形状一样，但它产出的不是内容，
  而是**音量旋钮**——过一遍 `silu` 激活后，每个维度一个 0 到 1 附近的数，
  决定 up 的内容"这一维放多少过去"；
- **逐元素相乘 $\odot$**：内容 × 旋钮，这就是"门"（gate 的本义）；
- **`down_proj`**（$3072 \to 1024$）：压回隐藏维度，和老 MLP 的 $W_2$ 一样。

`silu` 激活一句话：**SiLU(x) = x · sigmoid(x)**，一个平滑版的 ReLU——
负数不再一刀切归零，而是温柔地衰减（Qwen3 的 config.json 里
`"hidden_act": "silu"` 说的就是它）。

为什么这样更强？直觉版：老 MLP 的激活函数对"哪些信息该通过"是**死板的**
（GELU 曲线固定）；SwiGLU 让模型自己学习"此时此刻每维该开多大音量"，
门控本身是输入的函数。同样多的参数，表达力上了一个档次——
这也是 LLaMA、Qwen 等主流模型清一色用 SwiGLU 的原因。

### 实现 Qwen3MLP.forward

`__init__` 已给出（三个无 bias 的 Linear），`forward` 就是公式的直译：

```python
def forward(self, x):
    return self.down_proj(
        torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
    )
```

注意乘号 `*` 是**逐元素相乘**（两个 `(n, 3072)` 的张量对应位置相乘），
不是矩阵乘——这就是那道"门"。

> 💡 **你可能会问：三个矩阵比两个矩阵参数多，怎么还说"同样参数"？**
>
> 问得好。Qwen3-0.6B 的 `intermediate_size` 是 3072，是隐藏维 1024 的
> **3 倍**，而教学版 MLP 是 4 倍（`mlp_ratio=4`）。三个 3 倍宽的矩阵
> （$3 \times 1024 \times 3072$）和两个 4 倍宽的矩阵
> （$2 \times 1024 \times 4096$）参数量几乎一样—— SwiGLU 模型的中间维
> 就是按这个"等价交换"定的。

> 📌 **对标 vLLM：** 真实 vLLM 里 Qwen3 的实现在
> `vllm/model_executor/models/qwen3.py`（MLP 部分调用了共享的
> `vllm/model_executor/layers/mlp.py` 里的 `MergedParallelMLP`）。
> 你会看到 gate 和 up 在真实代码里常被**合并成一个大矩阵**一次算完再切开
> （少一次 kernel 调用），数学上和我们的三个 Linear 完全等价。

> 📌 **划重点：** RMSNorm = LayerNorm 砍掉"减均值"和 bias，
> $x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot w$；
> SwiGLU = down(silu(gate(x)) ⊙ up(x))，gate 是学出来的音量旋钮。

---

## 动手练习

1. 在 `minivllm/model/qwen3.py` 里实现 `RMSNorm.forward` 和
   `Qwen3MLP.forward`；
2. 本周没有给这两个模块单独的快测（它们的终极验收是第 6 章的
   真实权重对拍）。先写一个 30 秒的自检脚本确认没写歪：

```bash
.venv/bin/python - <<'EOF'
import torch
from minivllm.model.qwen3 import RMSNorm, Qwen3MLP, Qwen3Config

# RMSNorm：weight 全 1 时，输出的均方根应该约等于 1
norm = RMSNorm(8, 1e-6)
x = torch.randn(4, 8) * 100   # 故意放大 100 倍
y = norm(x)
rms = y.pow(2).mean(dim=-1).sqrt()
print("各行的 RMS：", rms)      # 应该全接近 1.0

# Qwen3MLP：形状对、能反向（参数都参与了计算）
cfg = Qwen3Config({"vocab_size": 100, "hidden_size": 16, "num_hidden_layers": 1,
                   "num_attention_heads": 2, "num_key_value_heads": 1,
                   "intermediate_size": 32, "rms_norm_eps": 1e-6})
mlp = Qwen3MLP(cfg)
out = mlp(torch.randn(5, 16))
print("MLP 输出形状：", out.shape)  # (5, 16)
out.sum().backward()
print("梯度流过所有矩阵：",
      all(p.grad is not None for p in mlp.parameters()))
EOF
```

3. （思考题）RMSNorm 的 weight 初始化成全 1。加载真实权重后它还是全 1 吗？
   想想训练会把这个向量学成什么样（提示：如果训练觉得某维该放大，
   它会学成大于 1 的数）。

## 参考答案

`reference/model/qwen3.py` 里的 `RMSNorm`（含 `forward` 共 10 行）和
`Qwen3MLP`（`forward` 只有 3 行）是标准答案。**卡住 20 分钟再看**，
重点对照：RMSNorm 的均值是不是沿 `dim=-1`、有没有 `.float()` 的来回转换；
SwiGLU 的乘号是逐元素的 `*` 而不是 `@`。

---

👉 下一章：[第 5 章：RoPE 与 GQA——位置编码与省内存](./05-RoPE与GQA-位置编码与省内存.md)
