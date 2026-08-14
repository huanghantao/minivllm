# 第 3 章：generate 与 stream——两种用法

> 本章你将：
> 1. 实现 `generate`：进一批、跑到完、按提交顺序返回 `list[RequestOutput]`；
> 2. 实现 `stream`：用生成器（`yield`）做打字机式的流式输出；
> 3. 掌握流式的核心算法——**增量文本 = 全文 decode 再减去已发前缀**，
>    并搞懂为什么不能逐 token decode；
> 4. 跑绿 `tests/test_w7.py` 的全部快测。

---

## 3.1 一台引擎，两种摇法

上一章造好了心跳 `step()`。但"摇手柄"的姿势，不同的用户想要两种：

- **批处理用户**（跑评测、洗数据）：我有一千条 prompt，你闷头全跑完，
  最后把一千份结果一起给我。中间过程我不看。——这是 `generate`；
- **聊天用户**（对着屏幕等人）：出一个词就给我显示一个词，
  别让我干瞪眼。——这是 `stream`。

有意思的是：这两种用法**不需要改引擎内核**，只是摇手柄的节奏和取件的方式不同。
这正是把 `while` 从函数肚子里挪出来的回报——控制权在调用方手里。

## 3.2 generate：攒齐再拿

先看代码（你要实现的）：

```python
def generate(self, prompts_token_ids, sampling_params=None):
    """同步生成：进一批、跑到完、按提交顺序返回 list[RequestOutput]。"""
    ids = [
        self.add_request(p, sampling_params) for p in prompts_token_ids
    ]
    while self.has_unfinished():
        self.step()
    return [self._make_output(self._requests[i]) for i in ids]
```

三行，每一行都有讲究：

1. **先进件**：每个 prompt 调一次 `add_request`，把返回的 request_id 按顺序
   收进 `ids`。注意 `add_request` 只入队不开工，所以这一行只是"把单子全收上来"；
2. **再摇到完**：`while self.has_unfinished(): self.step()`——就是 Week 5 那个
   `while` 循环，只是现在它是引擎外面的一层薄壳；
3. **最后按提交顺序取件**：`self._requests[i]` 从账本里按 id 翻出每个请求
   （第 1 章说过，账本只增不减，做完的也查得到），`_make_output` 把
   token id decode 成文本、包成 `RequestOutput`。

`_make_output` 已给出，就两行：

```python
def _make_output(self, req):
    text = self.tokenizer.decode(
        req.output_token_ids, skip_special_tokens=True
    )
    return RequestOutput(req, text)
```

`skip_special_tokens=True` 表示 decode 时把 `<|endoftext|>` 这类特殊符号
扔掉——用户不想在聊天回复里看到它们。

> 📌 **划重点：** `generate` 返回的顺序 = 提交顺序，不是完成顺序。
> 短请求可能先做完，但它的 `RequestOutput` 仍然排在自己该在的位置——
> 因为我们按 `ids`（提交时领的号）取件，而不是按下车的先后。
> 测试 `test_engine_batch_order_and_consistency` 专门断言了这一点。

## 3.3 stream：逐字吐出的生成器

`stream` 是 Python 的**生成器**（generator）——函数体里有 `yield`，
调用方用 `for piece in engine.stream(...)` 一片一片地取：

```python
def stream(self, prompt_token_ids, sampling_params=None):
    """流式生成（单请求）：每走一步，吐出新生成的文本增量。"""
    rid = self.add_request(prompt_token_ids, sampling_params)
    req = self._requests[rid]
    sent = 0
    while True:
        finished = self.step()
        text = self.tokenizer.decode(
            req.output_token_ids, skip_special_tokens=True
        )
        if len(text) > sent:
            yield text[sent:]
            sent = len(text)
        if rid in finished:
            break
```

注意三个设计：

1. **单请求**。`stream` 一次只服务一个 prompt（聊天场景就是一个用户在打字）。
   入队后立刻从账本里抓住 `req` 这个对象——之后每步都能从它身上读到最新的
   `output_token_ids`；
2. **增量算法**：每跳一步心跳，把**目前生成的全部 token** decode 成完整文本，
   和上次已发出的长度 `sent` 比一比，长出多少就吐多少（`text[sent:]`）。
   公式就一句话：**增量 = 全文 decode − 已发前缀**；
3. **下车判定**：`step()` 的返回值终于派上用场——`rid in finished` 说明
   本步之后这个请求做完了，`break` 收工。注意 `break` 在 `yield` **之后**：
   最后一步采出的词也要先吐给用户再走。

> 💡 **你可能会问：为什么不能每采一个 token 就 decode 这一个 token，
> 直接吐出去？非要每次从头 decode 全文？**
>
> 因为 **token ≠ 字符**，边界对不上。以 BPE 分词器为例：一个汉字可能由
> 两三个 token 拼成，一个 token 也可能是半个英文单词。逐 token decode，
> 你会看到一堆乱码碎片（半个字的字节根本 decode 不出来，会变成 ``）。
> 只有"把目前为止的所有 token 一起 decode"，分词器才能正确地拼字节。
> 从头 decode 看起来浪费，但生成的 token 通常就几百个，decode 一次的代价
> 相对一次模型前向可以忽略——**这是用一点点 CPU 换正确性**，非常划算。

为了让你亲眼看到这个差异，第 5 章实战里我们会打印 `stream` 吐出的每一片，
你会发现片的边界确实是"' Paris' → '.' → ' The' → …"这种**按词/标点**的节奏，
而不是按 token 的生硬切法。

> ⚠️ **易踩坑：** 三个高频错误——
> ① `yield text`（全量）而不是 `yield text[sent:]`（增量）——用户会看到
> 文本一遍遍地从头刷屏；
> ② `break` 写在 `yield` 前面——最后一片被吞掉，流式拼出的全文比
> `generate` 的结果少一截（`test_engine_stream_concat_equals_generate`
> 就是拿"所有片拼起来 == generate 的 text"来卡这个的）；
> ③ 用 `while self.has_unfinished()` 当循环条件——单请求场景下也能跑对，
> 但语义错了：`has_unfinished` 是"引擎里**任何**请求没做完"，万一以后引擎里
> 还跑着别的请求，这个 `stream` 就会陪着别人多摇。自己的请求下车就该走，
> 认准 `rid in finished`。

## 3.4 测试在钉哪些合同

`tests/test_w7.py` 的四个快测，正好把本章和上一章的行为钉死：

| 测试 | 钉住的合同 |
|---|---|
| `test_engine_generate_matches_single` | 引擎生成的 token 序列，和 Week 4 的 `paged_generate_single`（单请求手动分页生成）**一模一样**——换引擎不换结果 |
| `test_engine_batch_order_and_consistency` | 批量跑的结果 = 各自单独跑的结果，且按提交顺序返回 |
| `test_engine_stream_concat_equals_generate` | stream 所有片拼起来 == generate 的 text |
| `test_engine_stop_token` | 撞上 `stop_token_ids` 立即停车，`finish_reason == "stop"`，且停止词本身是最后一个词 |

第一个测试值得多看一眼：它用**另一套独立代码路径**（`tests/test_w4.py` 里的
`paged_generate_single`，不经过调度器、不经过引擎）算出标准答案，再断言你的
引擎给出同样的结果。这叫**对拍**——两套实现互相印证。你的引擎经过了
调度器准入、批量 decode、采样器这么一长串新代码，最终结果却和最朴素的
单请求路径分毫不差，这说明：引擎化**只改变了组织方式，没有改变计算结果**。

> 📌 **对标 vLLM：** 真实 vLLM 里 `generate` 对应 `LLM.generate`
> （`vllm/entrypoints/llm.py`），底层同样是一个"add_request + 循环 step
> 直到完成"的 `_run_engine` 循环；流式则走 `AsyncLLM.generate` 返回的
> 异步生成器（`vllm/v1/engine/async_llm.py`）。真实的增量 detokenize 更精细
> （有专门的 `detokenizer.py` 处理增量解码和特殊 token），但
> "decode 全文减前缀"的核心思想是一致的。

> 📌 **划重点：** `generate` = 进一批 + 摇到完 + 按号取件；
> `stream` = 单请求 + 每步"全文 decode 减已发前缀"地吐 + 自己下车就收工。

---

## 动手练习

1. 在 `minivllm/engine/llm_engine.py` 里实现 `LLMEngine.generate` 和
   `LLMEngine.stream`（先自己写，特别是 `sent` 那几行）；
2. 跑全部快测（应该 4 个全绿）：

```bash
.venv/bin/pytest tests/test_w7.py -m "not slow"
```

3. （思考题，不写代码）`generate` 的 `while` 循环跑在途中时，
   如果**另一个线程**调用了 `eng.add_request(...)`，会发生什么？
   这个中途插队的请求会被跑完吗？它的结果会出现在 `generate` 的返回列表里吗？
   （提示：看 `has_unfinished` 的判断范围和 `ids` 是什么时候定下来的。）

## 参考答案

`reference/engine/llm_engine.py` 里的 `generate` / `stream` 是标准答案，
加起来不到 25 行。**卡住 20 分钟再看**，重点对照：`sent` 的更新时机、
`yield` 和 `break` 的先后。

思考题的答案：插队请求**会被跑完**（`has_unfinished` 数的是所有未完成的请求，
`while` 循环会替它继续摇），但它的结果**不在**返回列表里（`ids` 进件时就定死了）。
这正是真实引擎的行为——引擎是公共的，谁的结果谁自己领走。

---

👉 下一章：[第 4 章：LLM 门面——一行起引擎](./04-LLM门面-一行起引擎.md)
