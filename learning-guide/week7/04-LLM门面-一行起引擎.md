# 第 4 章：LLM 门面——一行起引擎

> 本章你将：
> 1. 逐行读懂 `LLM.__init__`：一个模型名是怎么变成一台组装好的引擎的；
> 2. 实现 `_best_device()`：自动选 MPS 还是 CPU；
> 3. 实现 `LLM.generate` / `LLM.stream`：两个加起来不到五行的"翻译"方法；
> 4. 理解门面的意义：把"字符串世界"（用户）和"token 世界"（引擎）隔开。

---

## 4.1 为什么还需要一层壳

到上一章为止，`LLMEngine` 已经能干活了。但回头看看它的接口：

```python
eng.add_request(prompt_token_ids, sampling_params)   # 要 token id！
eng.generate(prompts_token_ids, sampling_params)     # 还是要 token id！
```

用户手里只有**字符串**和**一个模型名字**（`"Qwen/Qwen3-0.6B"`）。
从模型名到一台能跑的引擎，中间隔着一堆杂务：权重文件在哪、用什么设备、
什么精度、KV cache 开多大、分词器怎么建、结束符是谁……

这些杂务**每次用引擎都要做一遍，且做法固定**——典型的"门面"（facade）场景。
`LLM` 类就是这张脸，对标真实 vLLM 的用法：

```python
from minivllm.engine.llm import LLM
from minivllm.sampling.sampler import SamplingParams

llm = LLM(model_name="Qwen/Qwen3-0.6B")                 # 一行起引擎
outs = llm.generate(["The capital of France is"],       # 一行生成
                    SamplingParams(max_new_tokens=16))
print(outs[0].text)
```

## 4.2 __init__ 逐行读（已给出）

打开 `minivllm/engine/llm.py`，`__init__` 已给出，但每一行都值得讲：

```python
def __init__(
    self,
    model_name="Qwen/Qwen3-0.6B",
    max_model_len=2048,
    device=None,
    dtype=None,
    num_blocks=None,
    block_size=16,
    max_num_seqs=8,
):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    self.device = device or _best_device()
    if dtype is None:
        dtype = torch.bfloat16 if self.device == "mps" else torch.float32
    self.dtype = dtype
```

- **两个 import 写在函数里**而不是文件顶部：`huggingface_hub` 和 `transformers`
  都是重依赖，只有真的走 `LLM` 这条路（真实权重）才需要它们。写在里面，
  用 MiniTransformer 跑快测时就完全不用加载这两个库——快测才"快"；
- **设备**：用户没指定就调 `_best_device()` 自动选（这个函数归你写，见 4.3 节）；
- **精度**：MPS 上默认 `bfloat16`（Qwen3 官方权重就是这个精度，省一半内存），
  CPU 上退回 `float32`（CPU 对 bf16 的算子支持不全，fp32 最稳）。

接着是**定位权重**：

```python
    # 命中本地 HF 缓存；allow_patterns 避开缓存里可能缺失的 LICENSE 等无关文件
    model_path = snapshot_download(
        model_name,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    self.model = Qwen3ForCausalLM.from_pretrained(
        model_path, device=self.device, dtype=dtype
    )
    self.model.eval()
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
```

- `snapshot_download` 是 HuggingFace 的"下载器"：模型名 → 本地文件夹路径。
  好处是**命中缓存就秒回**——你 Week 6 对拍时权重已经下过，这里直接复用
  （离线时设 `HF_HUB_OFFLINE=1` 也能走缓存）；
- `allow_patterns` 只挑需要的文件（配置 json、权重 safetensors、词表 txt、
  聊天模板 jinja），跳过 LICENSE、README 之类——本地缓存里如果恰好缺这些
  无关文件，不过滤的话它会尝试联网补全，离线环境就报错了；
- `from_pretrained` 是你 Week 6 写的加载流程（读 config、建模型、按名字搬权重）；
- `model.eval()`：切到推理模式（关掉 dropout 之类的训练行为）；
- 分词器直接用 `transformers` 的 `AutoTokenizer`——分词器不属于本课程的
  手写范围，它是"输入设备"，借现成的。

最后是**配容量、组装引擎**：

```python
    # KV cache 容量：默认装得下 max_num_seqs 条 max_model_len 长的序列
    if num_blocks is None:
        total_tokens = max_model_len * max_num_seqs
        num_blocks = -(-total_tokens // block_size)  # 向上取整
    self.engine = LLMEngine(
        self.model,
        self.tokenizer,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        eos_token_id=self.tokenizer.eos_token_id,
    )
```

- **KV cache 开多大？** 默认策略：装得下"`max_num_seqs` 个并发 × 每条
  `max_model_len` 长"——比如默认 8 并发 × 2048 token = 16384 个槽位，
  除以块大小 16，向上取整就是 1024 个块。这是个**保守上界**：实际请求很少
  都顶到 max_model_len，所以物理块一般够用（真不够了，调度器的保守准入
  会让新请求排队，不会崩）；
- `-(-a // b)` 是"向上取整除法"的经典写法（负数地板除再取负），
  你在调度器的 `_blocks_needed` 里已经见过它；
- `eos_token_id` 从**分词器**身上取——结束符是模型的属性，写死在引擎里就错了。

> 💡 **你可能会问：`max_model_len` 只是个"配容量"的参数吗？
> 超过它的提示词会被拦住吗？**
>
> 在我们的 mini 版里，它**只用于算 KV cache 容量**，不做强制截断
> （真实 vLLM 会拒绝超长提示词）。提示词太长、块池装不下"提示词+全部待生成"时，
> 调度器的保守准入会让它一直在 waiting 里排队——不会报错，但也永远轮不到。
> 这是 mini 版的一个已知简化，用的时候心里有数即可。

## 4.3 轮到你了：三个小函数

`__init__` 已给出，你要填的是三个"小翻译"：

**① `_best_device()`**——一句话的设备选择：

```python
def _best_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"
```

`torch.backends.mps.is_available()` 是 PyTorch 自带的探针：在 Apple Silicon
的 Mac 上返回 `True`。有 MPS 走 MPS（GPU 加速），没有就 CPU 兜底。

**② `LLM.generate`**——字符串进，`RequestOutput` 出：

```python
def generate(self, prompts, sampling_params=None):
    """批量生成。prompts 是字符串或字符串列表；返回 list[RequestOutput]。"""
    return self.engine.generate(self._tokenize(prompts), sampling_params)
```

**③ `LLM.stream`**——单个字符串进，逐片吐文本：

```python
def stream(self, prompt, sampling_params=None):
    """流式生成单个 prompt，逐步吐出文本增量。"""
    (ids,) = self._tokenize(prompt)
    yield from self.engine.stream(ids, sampling_params)
```

两个方法都靠已给出的 `_tokenize` 做翻译：

```python
def _tokenize(self, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    return [self.tokenizer.encode(p) for p in prompts]
```

它顺手做了个贴心事：传单个字符串也认（包成单元素列表），省得用户记
"`generate` 必须传列表"。`stream` 里的 `(ids,) = ...` 是"解包一个元素的列表"
的写法——stream 只服务单请求，直接把那唯一的一份 token id 拆出来。
`yield from` 则是"把内层生成器的每一片原样转发出去"。

> ⚠️ **易踩坑：** `LLM.generate` 里**不要**自己 `encode` 完再调
> `engine.add_request` 手动摇 `step`——引擎已经有现成的 `generate`，
> 门面只做翻译、不做重复劳动。壳越薄越好：所有"怎么跑"的逻辑都留在引擎里，
> 将来改引擎（比如 Week 8 加测量）只改一处。

## 4.4 门面一隔，两个世界

到这儿，整个调用链终于闭环了：

```
用户（字符串、模型名）
  └── LLM          ← 翻译层：字符串 ↔ token id，模型名 → 组装好的引擎
        └── LLMEngine   ← 调度层：收件箱、心跳、账本
              └── Scheduler / BlockPool / PagedKVCache / Sampler / Model
                        ← 零件层：你 Week 4/5/6 的手工作品
```

用户从此不需要知道 token 是什么、块表是什么、调度器几点点名——就像开车的人
不需要懂变速箱。而你懂，因为变速箱是你亲手装的。

> 📌 **对标 vLLM：** 这一层对应真实 vLLM 的 `vllm/entrypoints/llm.py`
> 里的 `LLM` 类——构造时同样做"解析模型名 → 加载 → 建分词器 → 建引擎"，
> 对外暴露 `generate` / `chat` 等字符串级 API。真实版的构造参数多得多
> （tensor 并行、量化、显存占用比……），但"门面只做翻译和组装"的职责一模一样。

> 📌 **划重点：** `LLM` = 翻译 + 组装。`_tokenize` 把字符串翻成 token id，
> `__init__` 把模型名组装成引擎，两个 `generate`/`stream` 方法是薄得不能再薄的
> 转发。重活在引擎里，不在壳上。

---

## 动手练习

1. 在 `minivllm/engine/llm.py` 里实现 `_best_device`、`LLM.generate`、
   `LLM.stream`（都是小函数，先自己写）；
2. 这三个函数只有走真实模型才会被调到，验证靠 slow 测试（下一章正式跑）：

```bash
.venv/bin/pytest tests/test_w7.py -m slow -k llm_facade
```

（第一次跑要加载 Qwen3-0.6B 权重，耐心等一两分钟。）
3. （思考题，不写代码）`LLM.stream` 为什么用 `yield from` 而不是
   `return self.engine.stream(...)`？两种写法调用方拿到的对象有什么差别？
   （提示：一个是生成器，一个是"生成器的生成器"。）

## 参考答案

`reference/engine/llm.py` 是标准答案，整个文件只有 80 行。**卡住 20 分钟再看**。
思考题答案：`yield from` 让 `LLM.stream` 本身就是生成器，`for p in llm.stream(...)`
直接拿到每一片文本；而 `return` 版本里 `llm.stream(...)` 返回的是一个生成器，
调用方得再多包一层才能用——`yield from` 把这一层抹平了。

---

👉 下一章：[第 5 章：实战——多人同时聊天](./05-实战-多人同时聊天.md)
