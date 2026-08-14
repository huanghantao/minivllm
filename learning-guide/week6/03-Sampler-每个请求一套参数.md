# 第 3 章：Sampler——每个请求一套参数

> 本章你将：
> 1. 认识 `SamplingParams`——一张"采样参数表"，把挑词策略的全部旋钮装进一个对象；
> 2. 把前两章的零件串成完整流水线 `sample_token`：
>    贪心分支 → 温度 → top-k → top-p → softmax → 抽签；
> 3. 搞懂 `seed` 为什么能让随机采样"可复现"，以及为什么要**每个请求一个随机数发生器**；
> 4. 读完已给出的 `Sampler` 类——它是引擎（Week 7）直接调用的接口；
> 5. 跑绿本周前半的全部测试：`pytest tests/test_w6_sampler.py`。

---

## 3.1 SamplingParams：一张参数表

真实场景里，每个用户想要的生成风格都不一样：客服机器人要稳（低温），
创意写作要野（高温），代码补全要死板（贪心）。引擎不能为每种风格改代码，
而是让**每个请求自带一张参数表**。这就是 `SamplingParams`：

```python
@dataclass
class SamplingParams:
    """一次生成请求的采样参数表。

    temperature = 0 表示贪心（greedy），此时 top_k / top_p 不生效。
    """

    temperature: float = 0.0
    top_k: int = -1          # -1 表示不截断
    top_p: float = 1.0       # 1.0 表示不截断
    max_new_tokens: int = 16
    stop_token_ids: tuple = field(default_factory=tuple)
    seed: int = None         # 给了 seed，采样结果可复现
```

六个字段分两组：

| 字段 | 管什么 | 本周/以后 |
|---|---|---|
| `temperature` | 胆子大小；**0 = 贪心暗号** | 本周 |
| `top_k` / `top_p` | 画圈方式；-1 和 1.0 都表示"不画圈" | 本周 |
| `seed` | 随机种子，给了就能复现 | 本周 |
| `max_new_tokens` / `stop_token_ids` | 生成多少、遇到谁停 | Week 7 引擎读它们 |

注意默认值的设计：**全默认值 = 纯贪心、生成 16 个词**。也就是说不传任何参数，
行为和前五周一模一样——老代码零改动兼容。

> 📌 **对标 vLLM：** 这就是 `vllm/sampling_params.py` 的 mini 版。
> 真实的 `SamplingParams` 有三十多个字段（presence_penalty、logprobs、
> beam search……），但最核心的就是这几个。学完本节你打开那个文件，
> 前 50 行全能看懂。

## 3.2 sample_token：把零件串成流水线

现在把第 1、2 章的零件组装成一条完整的采样流水线。合同：

```python
def sample_token(logits, params=None, generator=None):
    """完整采样流水线：temperature → top_k → top_p → 按概率抽签。

    logits: (vocab,)；params 为 None 时等价贪心。
    params.temperature == 0 时走贪心（不看 top_k / top_p）。
    返回 token id（int）。
    """
```

参考答案的全貌（先自己写！）：

```python
params = params or SamplingParams()
if params.temperature == 0:
    return greedy_sample(logits)
logits = apply_temperature(logits.float(), params.temperature)
logits = top_k_filter(logits, params.top_k)
logits = top_p_filter(logits, params.top_p)
probs = torch.softmax(logits, dim=-1)
return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
```

逐行看：

1. **`params or SamplingParams()`**：不传参数？给你一份全默认的——即贪心；
2. **贪心分支**：`temperature == 0` 直接 argmax 返回，**根本不看 top_k/top_p**。
   这是测试 `test_sample_token_greedy_ignores_filters` 钉死的合同：
   哪怕 params 里写了 `top_k=1, top_p=0.01`，贪心照样只挑第一名；
3. **温度 → top-k → top-p**：顺序不能乱。先调温度改变分布形状，
   再画圈砍人（`.float()` 是把 logits 转成 float32，避免半精度下
   除法丢精度——这是个防御性细节）；
4. **softmax 后 `torch.multinomial`**：新操作一句话——`multinomial` 按概率
   分布**抽签**，`num_samples=1` 抽一张，返回中奖词的下标。
   这是"随机性"真正进来的地方，之前所有步骤都只是准备彩票。

> ⚠️ **易踩坑：** 别用 `torch.argmax(probs)` 代替 `multinomial`——那又退回
> 贪心了；也别用"生成 0~1 随机数再手动找区间"的土办法，`multinomial`
> 一行搞定且数值更稳。另外三个 filter 的顺序写反（比如先 top-p 再温度），
> 结果会不一样：温度会改变概率分布，直接影响 top-p 画多大的圈。

## 3.3 seed：让随机"可复现"

随机采样有个麻烦：这次输出一段好诗，下次同样的输入却变了样——
**没法复现，就没法调试、没法写测试。**

解决办法是**随机种子（seed）**。计算机里的"随机"其实是伪随机：
一个确定性的数列发生器，种子相同则数列完全相同。PyTorch 里这个发生器
叫 `torch.Generator`：

```python
g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
# 用 g1 和 g2 分别抽签，抽到的序列一模一样
```

测试 `test_sample_token_reproducible_with_seed` 就是这么验证的：
两个种子都是 42 的 `Generator`，各采 5 个词，结果列表逐项相等。

> 💡 **你可能会问：为什么不直接 `torch.manual_seed(42)` 设全局种子？**
>
> 全局种子只有**一个**，而引擎同时服务多个请求。请求 A 采一个词，
> 全局发生器就被消耗一次；A 的下一个词抽到什，取决于"在它之前
> 有多少别的请求抽过签"——顺序一乱，A 的结果就变了。
> 解法是**每个请求发一个独立的 Generator**（种子是它的 seed），
> 互不干扰，单个请求重来多少次结果都一样。这正是 `Sampler` 类干的事。

## 3.4 Sampler 类：引擎的采样前台（已给出，讲读）

`Sampler` 已经帮你写好了，它只比裸函数多管一件事：**给每个请求保管
一个随机数发生器**。

```python
class Sampler:
    """给引擎用的采样器：一次服务一批请求，各自有各自的参数。"""

    def __init__(self):
        # request_id -> torch.Generator（有 seed 的请求才有）
        self._generators = {}

    def _generator_for(self, request_id, seed, device):
        if seed is None:
            return None
        if request_id not in self._generators:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            self._generators[request_id] = g
        return self._generators[request_id]

    def sample_one(self, logits, request_id, params):
        """给单个请求采一个 token。logits: (vocab,)"""
        g = self._generator_for(request_id, params.seed, logits.device)
        if g is not None:
            logits = logits.cpu()
            token = sample_token(logits, params, generator=g)
        else:
            token = sample_token(logits, params)
        return token

    def drop_request(self, request_id):
        """请求结束后清掉它的随机数发生器。"""
        self._generators.pop(request_id, None)
```

三个细节值得看懂：

1. **懒建发生器**：`_generator_for` 第一次见到某 `request_id` 才建
   `Generator` 并设种子，之后一直复用同一个——同一个请求的第 1 步、
   第 2 步……消耗的是同一条伪随机数列，这正是"可复现"的来源；
2. **为什么要 `logits.cpu()`**：`Generator` 建在 CPU 上，而 `multinomial`
   要求概率张量和发生器在**同一台设备**。模型可能跑在 MPS（Mac 显卡）上，
   所以带 seed 的采样要把 logits 搬回 CPU 再抽。不带 seed 的请求无所谓，
   直接用全局随机源，在哪抽都行；
3. **`drop_request` 清理**：请求结束了就把它的发生器删掉，不然 `_generators`
   字典越攒越大（内存泄漏），而且同名请求再来时会继承旧的随机数进度，
   复现性就破了。测试 `test_sampler_per_request_seed` 验证的正是这件事：
   `drop_request(1)` 之后重新采样，发生器重建、种子重置，结果和第一次相同。

Week 7 的引擎每走一步，就是对 batch 里每个请求各调一次 `sample_one`；
请求完结束时调一次 `drop_request`。接口就这么大。

## 3.5 真实世界的参数长什么样

看一眼 Qwen3-0.6B 官方模型目录里的 `generation_config.json`
（第 7 章会完整巡礼这个目录）：

```json
{
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95
}
```

官方推荐不是贪心，而是**低温 + 双圈**：0.6 让分布稍微变尖（比 1.0 稳），
top_k=20 砍掉长尾垃圾，top_p=0.95 再自适应收一道。你现在完全看得懂
这三个数字各自在干什么了——它们就是你这两天手写的三个函数。

> 📌 **划重点：** `sample_token` 流水线 = 贪心分支 → 温度 → top-k → top-p →
> softmax → `multinomial` 抽签；`seed` 通过**每请求一个 Generator** 实现
> 可复现；`Sampler` 负责保管这些发生器，请求结束要 `drop_request`。

---

## 动手练习

1. 在 `minivllm/sampling/sampler.py` 里实现 `sample_token`
   （照 3.2 节的流水线，注意分支顺序）；
2. 跑本周前半全部测试：

```bash
.venv/bin/pytest tests/test_w6_sampler.py
```

   9 条用例应该全绿，包括之前红着的
   `test_sample_token_greedy_ignores_filters` 和
   `test_sample_token_respects_top_k_support`。
3. （思考题，不写代码）`test_sample_token_respects_top_k_support` 里，
   `logits[3] = 100`、`logits[7] = 99`，`top_k=2` 采 20 次。
   为什么断言只需要检查"采中的词 ∈ {3, 7}"，而不用管 3 和 7 各中了几次？
   提示：想想 100 和 99 过完 softmax 各分多少概率。

## 参考答案

`reference/sampling/sampler.py` 里的 `sample_token` 是标准答案，7 行。
**卡住 20 分钟再看**，重点对照：贪心分支有没有放在最前面、
三个 filter 的顺序、最后是不是 `multinomial` 而不是 `argmax`。

---

👉 下一章：[第 4 章：Qwen3 的四处升级——RMSNorm 与 SwiGLU](./04-Qwen3的四处升级-RMSNorm与SwiGLU.md)
