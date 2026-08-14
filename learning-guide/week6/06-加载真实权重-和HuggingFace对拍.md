# 第 6 章：加载真实权重——和 HuggingFace 对拍

> 本章你将：
> 1. 逐段读懂 `from_pretrained`（已给出）：一个 1.4 GB 的模型目录是怎么灌进我们的 `nn.Module` 的；
> 2. 看懂权重键名的"对号入座"规则——为什么我们的模块名和 HF 一模一样；
> 3. 亲手实现三条前向路径：`naive_forward`（整段）、`prefill`、`decode_batch`；
> 4. 逐行读懂三条对拍测试：logits 偏差 < 5e-3、贪心生成逐词一致、分页接口 == 朴素前向；
> 5. 跑绿 `pytest tests/test_w6_qwen3.py -m slow`——本周的金牌时刻。

---

## 6.1 对拍：我们的模型 vs 官方模型

前两周我们把引擎的分页、调度都做出来了，但一直缺一块硬证据：

> 我们的 Transformer 实现，和 HuggingFace 官方实现，**数学上是不是同一个模型？**

本周给出证据的方式叫**对拍**：同一个输入，两边各算一遍，逐数字比。
对拍能过，意味着此后我们的引擎说出的每一个字都和官方一致——
引擎剩下的差异只在"快不快"，不在"对不对"。

对拍的前提是：把官方训好的权重**原样**搬进我们的模型。这就是
`from_pretrained` 干的事。

## 6.2 from_pretrained 逐段讲（已给出）

模型目录里和对拍有关的是两个文件：`config.json`（模型的"身份证"，
写着 28 层、16 头这些身材数据）和 `model.safetensors`（权重本体，
约 1.4 GB）。`from_pretrained` 分五步把它们变成我们的
`Qwen3ForCausalLM`：

```python
@classmethod
def from_pretrained(cls, model_path, device="cpu", dtype=torch.float32):
    """从本地 HF checkpoint 目录加载（config.json + model.safetensors）。"""
    from safetensors.torch import load_file

    with open(os.path.join(model_path, "config.json")) as f:
        config = Qwen3Config(json.load(f))
    model = cls(config)
    state = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            state.update(load_file(os.path.join(model_path, fname)))
    # HF 的键以 "model." 开头，我们的模块就是按这个结构命名的，
    # 剥掉前缀即可对上；lm_head 与嵌入共享权重，checkpoint 里没有它。
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if k != "lm_head.weight"]
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not unexpected, f"checkpoint 里多了不认识的权重: {unexpected}"
    assert not missing, f"checkpoint 缺了权重: {missing}"
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    return model.to(device=device, dtype=dtype)
```

**第一步：读身份证。** `config.json` 是 JSON 文本，`Qwen3Config` 把里面的
字段翻译成我们习惯的属性名（`num_hidden_layers` → `num_layers` 等）。

**第二步：建空壳模型。** `cls(config)` 按配置搭好全部模块——此时权重是
随机初始化的，形状已对、数值全错。

**第三步：读权重本体。** safetensors 文件就是一个"名字 → 张量"的大字典
（第 7 章细聊这个格式），`load_file` 读进来合并成 `state`。
Qwen3-0.6B 共 311 个张量、约 7.5 亿个参数。

**第四步：键名对号入座。** HF 的键长这样（第 0 层）：

| checkpoint 里的键 | 我们的模块 |
|---|---|
| `model.embed_tokens.weight` | `embed_tokens.weight` |
| `model.layers.0.self_attn.q_proj.weight` | `layers[0].self_attn.q_proj.weight` |
| `model.layers.0.self_attn.q_norm.weight` | `layers[0].self_attn.q_norm.weight` |
| `model.layers.0.mlp.gate_proj.weight` | `layers[0].mlp.gate_proj.weight` |
| `model.layers.0.input_layernorm.weight` | `layers[0].input_layernorm.weight` |
| `model.norm.weight` | `norm.weight` |

发现了吗？**剥掉 `model.` 前缀后逐字符相同**——这不是巧合，是我们定义
类时刻意照抄了 HF 的模块命名。这就是 `removeprefix("model.")` 那一行的
全部工作。命名对齐是权重加载里最重要的工程习惯：名字对了，
`load_state_dict` 自动完成几百个张量的对号入座。

**第五步：豁免 lm_head + 绑定。** `strict=False` 加载后，代码检查
`missing`（模型有、checkpoint 没有）和 `unexpected`（checkpoint 有、
模型没有）两份名单，**唯一允许出现的名字是 `lm_head.weight`**。
为什么？Qwen3-0.6B 开了 `tie_word_embeddings`：输出头 lm_head 和输入嵌入
**共享同一块矩阵**（第 7 章讲为什么合理），所以我们的模型里
`lm_head.weight` 不是一个独立参数，加载后手动指过去：

```python
model.lm_head.weight = model.embed_tokens.weight
```

最后 `.to(device=device, dtype=dtype)` 搬家，收工。

> 💡 **你可能会问：键名万一没对齐会怎样？**
>
> 那两条 `assert` 就是防线：多一个、少一个键都会立刻炸出来，
> 把名单打印给你看。**静默错位**（形状碰巧一样的张量装错位置）才是
> 权重加载最可怕的 bug——模型能跑，输出全是乱码还没报错。
> 所以对拍测试才是真正的终审。

## 6.3 你要实现的三条前向路径

`from_pretrained` 把权重灌进来了，但前向计算的代码要你自己写。
三条路径，其实是**同一份数学的三种包装**：

### 路径一：naive_forward——整段一次算完（对拍专用）

这是 Week 2 的老朋友，不接 cache，一条序列从头算到尾。
`Qwen3Attention.naive_forward` 的流程（`x: (L, E)` → `(L, E)`）：

1. `_project_qkv(x)`（已给出）：三个投影 + QK-Norm，
   得 `q: (L, 16, 128)`、`k/v: (L, 8, 128)`；
2. `apply_rotary_pos_emb` 给 q、k 上旋转（**v 不转**，上周的坑）；
3. `_repeat_kv` 把 k/v 扩成 `(L, 16, 128)`；
4. `permute` 换维成 `(1, H, L, D)`，套上 Week 2 的因果掩码，
   喂给 `scaled_dot_product_attention`（Week 2 写的，原样复用！）；
5. 输出 reshape 回 `(L, 16×128)`，过 `o_proj`。

`Qwen3DecoderLayer.naive_forward` 和 `Qwen3ForCausalLM.forward`
则是纯粹的组装，和 MiniTransformer 一模一样的骨架：

```python
# 一层：两个残差连接，norm 在前面（pre-norm）
x = x + self.self_attn.naive_forward(self.input_layernorm(x), cos, sin)
x = x + self.mlp(self.post_attention_layernorm(x))

# 整个模型：嵌入 → 28 层 → 最终 RMSNorm → lm_head
x = self.embed_tokens(input_ids)
for layer in self.layers:
    x = layer.naive_forward(x, cos, sin)
return self.lm_head(self.norm(x)).unsqueeze(0)
```

### 路径二、三：prefill / decode_batch——接上分页 cache

这两个是 Week 4 在 MiniTransformer 上练过的引擎接口，签名一字不差。
逻辑和 naive 几乎相同，唯一区别：**K/V 不现算现用，而是写进
`PagedKVCache` 再 gather 回来**（Attention 层的 prefill/decode_batch 里，
先 `kv_cache.write(...)` 写入本步的 k/v，再 `gather` / `gather_padded`
读回全部历史，然后 repeat、算注意力）。

有一个 Week 4 没有的新细节：RoPE 的 cos/sin 要转成和权重一样的 dtype
（模型可能是 bf16，而 cos/sin 是 float32 现算的）：

```python
cos, sin = self.rotary_emb(positions)
cos = cos.to(self.embed_tokens.weight.dtype)
sin = sin.to(self.embed_tokens.weight.dtype)
```

> ⚠️ **易踩坑：** decode_batch 里 `gather_padded` 返回的 k/v 形状是
> `(B, Lmax, Hkv, D)`，先 `transpose(1, 2)` 换成 `(B, Hkv, Lmax, D)`
> **再** `_repeat_kv`（复制的是头维）。顺序反了会把序列维当头维复制，
> 形状直接爆炸。

## 6.4 三条对拍测试逐行讲

`tests/test_w6_qwen3.py` 全部标记 `slow`（要加载真实权重，几十秒）。
三条测试，一条比一条硬核。

### 测试一：logits 几乎一致

```python
def test_logits_match_hf(our_model, hf_model, prompt_ids):
    with torch.no_grad():
        ours = our_model(prompt_ids.unsqueeze(0))[0]
        theirs = hf_model(prompt_ids.unsqueeze(0)).logits[0]
    diff = (ours - theirs).abs().max().item()
    assert diff < 5e-3, f"logits 最大偏差 {diff}"
```

- 同一句 `"The capital of France is"`，两边各算一遍，取最大偏差；
- **为什么阈值是 5e-3 而不是 0？** 浮点加法不满足结合律，我们矩阵乘的
  累加顺序和 HF 不可能完全相同，误差在 28 层里累积。对 15 万维、
  数值范围几十的 logits 来说，5e-3 的差异如同身高差一根头发——
  纯属数值噪声，不是实现错误。

### 测试二：贪心生成逐词一致

```python
def test_greedy_continuation_matches_hf(our_model, hf_model, prompt_ids):
    ours = generate_mod.generate_naive(
        our_model, prompt_ids.unsqueeze(0), max_new_tokens=8
    )[0, prompt_ids.shape[0]:].tolist()
    out = hf_model.generate(
        prompt_ids.unsqueeze(0), max_new_tokens=8, do_sample=False,
        pad_token_id=0,
    )
    theirs = out[0, prompt_ids.shape[0]:].tolist()
    assert ours == theirs
```

这条比上一条更"讲道理"：logits 差一点点没关系，只要 **argmax 选出的词
一样**，生成结果就逐词一致。`do_sample=False` 是让 HF 也走贪心。
用 Week 1 写的 `generate_naive` 驱动我们的模型——你看，连生成循环
都是亲手写的旧代码。

### 测试三：分页接口 == 朴素前向

```python
def test_paged_path_matches_naive(our_model, prompt_ids):
    pool = pool_mod.BlockPool(num_blocks=64, block_size=16)
    kv = paged_mod.PagedKVCache(
        C.num_layers, C.num_kv_heads, C.head_dim, num_blocks=64, block_size=16
    )
    ...
    logits = our_model.prefill(prompt_ids, kv, slots, bt)
    assert torch.allclose(logits, ref, atol=1e-4)
```

这是 Week 4"铁证"测试的**真实模型版**：同一个 Qwen3，走分页 cache 的
prefill + decode_batch，和整段 naive 前向逐步对拍——prefill 的末位
logits 用 `allclose` 比，decode 两步逐词比 argmax。它证明：我们为速度
发明的所有机制（分页、写读 cache、补齐头数），**不改变模型的任何一个字**。

> 📌 **划重点：** 对拍三板斧——logits 数值偏差 < 5e-3（容忍浮点噪声）、
> 贪心 8 词逐字一致（argmax 一致才是真一致）、分页路径 == 朴素路径
> （工程优化不改变语义）。三条全绿 = 你的 Qwen3 和官方是同一个模型。

## 6.5 跑起来

```bash
# 先看参考答案全绿（几十秒，加载 1.4 GB 权重）
IMPL=reference .venv/bin/pytest tests/test_w6_qwen3.py -m slow

# 再打自己的实现
.venv/bin/pytest tests/test_w6_qwen3.py -m slow
```

权重读的是本地 HF 缓存；网络不稳可以加 `HF_HUB_OFFLINE=1` 强制离线。
全绿之后不妨亲手玩一把——把 prompt 换成中文，看看你的引擎
（随机权重乱码了五周之后）第一次说出通顺的人话。

> 📌 **对标 vLLM：** 真实 vLLM 的权重加载在
> `vllm/model_executor/model_loader/`（支持 safetensors、gguf、
> 量化格式等十几种来源），配置解析在 `vllm/transformers_utils/config.py`。
> 思路和我们的 `from_pretrained` 完全一样：读 config → 建模型骨架 →
> 按键名对号入座 → 处理 tied weights——只是它要兼容几百种模型，
> 所以多了一层"模型注册表"。

---

## 动手练习

1. 在 `minivllm/model/qwen3.py` 里实现：`Qwen3Attention.naive_forward`、
   `Qwen3DecoderLayer.naive_forward`、`Qwen3ForCausalLM.forward`
   （先做路径一，跑通前两条对拍）；
2. 再实现 `Qwen3Attention.prefill / decode_batch`、
   `Qwen3DecoderLayer.prefill / decode_batch`、
   `Qwen3ForCausalLM.prefill / decode_batch`（路径二三，跑通第三条对拍）；
3. 跑：

```bash
.venv/bin/pytest tests/test_w6_qwen3.py -m slow
```

4. （思考题）`test_paged_path_matches_naive` 里 `PagedKVCache` 的头数参数
   传的是 `C.num_kv_heads`（8）而不是 `C.num_heads`（16）。结合第 5 章，
   说说为什么——cache 里存的 K/V 是复制前还是复制后的？

## 参考答案

`reference/model/qwen3.py` 是全部三条路径的标准答案。**卡住 20 分钟
再看**，建议顺序：先让 `naive_forward` 链路对上（前两条测试），
再把 Week 4 的分页经验搬过来写后两条。重点对照：RoPE 有没有只转
q/k、`_repeat_kv` 在 gather 之后调用、decode 里的 transpose 顺序。

---

👉 下一章：[第 7 章：AI 联系——模型文件里都装了什么](./07-AI联系-模型文件里都装了什么.md)
