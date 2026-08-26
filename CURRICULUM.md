# 课程大纲与写作说明书（给教程子 agent 的作业手册）

> 本文档是《minivllm：从零手写 LLM 推理引擎》的**总纲**。每周由一个独立的子 agent 负责撰写，
> 请严格遵守本文档的约定，保证 8 周风格一致、章节之间严丝合缝。

---

## 0. 一句话课程定位

参考真实 vLLM 源码（pip 安装 `vllm==0.27.1`，源码就在 `.venv` 的 site-packages 里），用 PyTorch 从零手写一个
**能加载 Qwen3-0.6B 真实权重、带分页 KV cache 和 continuous batching 的迷你推理引擎**，
全程在 macOS（Apple Silicon，MPS）上可跑。

## 1. 读者画像（写作前必读，是真人实测的）

- **Python：比较熟。** 类、模块、pip 都会。numpy 见过一点。
- **PyTorch：基本没用过。** Week 1 第 2 章专门热身（tensor 创建、索引、矩阵乘、device）。
  之后每个新 torch 操作出现时都要一句话解释（view/transpose/permute/softmax/argmax/cat…）。
- **Transformer：比较模糊。** 大概知道"注意力"，公式记不清。Week 2 从零重建，
  不许假设读者记得 Q/K/V。
- **LLM 推理：没概念。** Week 1 必须从"下一个词预测"讲起，prefill/decode 这些词第一次出现
  都要给定义。
- **操作系统：完全不知道。** Week 4 的分页思想必须用生活类比从零建立（储物柜/停车场），
  "虚拟内存""页表"这些词要靠类比落地。
- **数学：弱。** 向量/矩阵乘法用到时一句话带复习（"对应相乘再相加"）；不出现任何证明；
  softmax 第一次出现时给公式 + 直觉。
- **学习动机：读懂 vLLM 源码 + 亲手写出能跑真实模型的推理引擎。**
  每周都要有"这和真实 vLLM 的对应关系"（📌 对标 vLLM 盒子）。
- **每周投入 7 小时以上**，每章可以写长、配练习。

### 结业目标（第 8 周大结业围绕它们设计）

1. 能手写加载 Qwen3-0.6B 真实权重的推理引擎（分页 KV cache + continuous batching +
   采样 + 流式输出），并说出每个模块在解决什么问题；
2. 能打开真实 vllm v1 的 `scheduler.py` / `block_pool.py` / `llm_engine.py` 不慌，
   认得出每个核心概念；
3. 讲得清 PagedAttention 和 continuous batching 各自解决了什么浪费。

## 2. 仓库布局（写章节时引用的路径）

```
minivllm/
├── learning-guide/weekN/   ← 你写的东西在这里（教程正文）
├── minivllm/               ← 读者的战场：函数体 raise NotImplementedError("TODO")，读者自己填
├── reference/              ← 参考答案：完整实现（已定稿、已通过全部测试）
├── tests/                  ← 测试（已定稿：58 个快测 + 4 个慢测，reference 全绿）
├── figures/gen_*.py        ← 生图脚本（已定稿，图片已生成在 figures/out/）
├── figures/out/*.png       ← 教程里引用的图片
├── figures/data/bench.json ← 实测性能数据（写作时引用这些真实数字）
├── scripts/                ← 基准实测脚本（已定稿）
└── material/vllm-源码对照.md ← Week 8 的阅读材料
```

**代码与测试已经全部定稿并验证通过，子 agent 只写教程，不要改代码、测试、图片脚本。**
如果发现代码确有 bug，在交付说明里报告，不要自行修改。

### 测试机制（写"动手练习"时必须引用对）

- 测试用 IMPL 环境变量选择打哪个包：默认打 `minivllm`（读者的战场），
  `IMPL=reference` 打参考答案。
- 读者第一周跑 `IMPL=reference .venv/bin/pytest tests` 看全绿（见证终点），
  再跑 `.venv/bin/pytest tests` 看红（战场现状）。
- 每周的练习收尾命令：`.venv/bin/pytest tests/test_wN.py`（本周文件变绿）。
- 慢测试（要加载真实 Qwen3-0.6B 权重）：`pytest tests -m slow`，
  权重已在本地 HF 缓存（离线可用 `HF_HUB_OFFLINE=1`）。

## 3. 写作风格（统一要求）

1. **口语化、说人话。** 像一位耐心的朋友在讲解。允许"你可能会想…""慢着！"这类对话感。
2. **每个概念：生活类比 → 图形直觉 → 严格定义 → 代码 → 验证。** 类比先行，定义殿后。
3. **多用表格**总结对比、多用编号步骤。
4. **固定盒子**：
   - `> 📌 对标 vLLM：`——本节内容在真实 vLLM 源码里的对应（给出具体文件路径）；
   - `> 💡 你可能会问：`——预判读者困惑并解答；
   - `> ⚠️ 易踩坑：`——常见错误；
   - `> 📌 划重点：`——一句话总结本节。
5. **不许要求读者证明任何东西。** 只建立直觉。
6. **数学公式用 LaTeX**（mdBook 通过 `theme/head.hbs` 加载 jsdelivr 的 MathJax 3）：行内 `$...$`、独立行 `$$...$$`；
   **公式内换行必须写 `\\\\`（四个反斜杠）**——mdBook 的 Markdown 渲染会吃掉一层反斜杠。
   **代码块与行内代码中的内容绝不放 `$`。**
7. **每章开头**放一行 `> 本章你将：` 的导读，列出 3-5 个收获。
8. **每章结尾固定小节**：
   - `## 动手练习`：1-3 个练习（填 `minivllm/xxx.py` 里的函数 + 跑测试命令）；
   - `## 参考答案`：一句话指向 `reference/` 同名文件，提示"卡住 20 分钟再看"。
9. **代码引用必须真实**：所有函数签名、行为以 `reference/` 为准（见第 6 节 API 速查）。
   教程里出现的每一段代码都必须能在仓库里跑通，不许凭印象写 API。

## 4. 图片政策（硬性要求）

- **凡涉及精确坐标、几何位置的图，一律用仓库里已生成的 PNG**，**禁止 ASCII 图**。
- 图片引用方式（章节在 `learning-guide/weekN/` 下）：
  `![配图说明](../../figures/out/w4_block_table.png)`
- 现有图片清单（`figures/out/`）：

| 文件 | 内容 | 归属周 |
|---|---|---|
| w1_autoregressive.png | 自回归流程：新词接回输入 | W1 |
| w1_benchmark.png | 实测：naive 39 vs vLLM 146 tok/s | W1 |
| w2_transformer_map.png | MiniTransformer 结构图 | W2 |
| w2_attention_heatmap.png | 真实注意力权重热力图（因果掩码） | W2 |
| w3_recompute_waste.png | naive 每步重算整段的浪费 | W3 |
| w3_speedup.png | 实测：KV cache 快 3.0 倍 | W3 |
| w3_memory_bill.png | KV cache 内存账（Qwen3-0.6B） | W3 |
| w4_fragmentation.png | 连续分配的碎片 vs 分页 | W4 |
| w4_block_table.png | 逻辑 token → 物理块映射 | W4 |
| w5_gantt.png | 静态批处理 vs continuous batching 甘特图 | W5 |
| w5_throughput.png | 实测：批处理吞吐 19.6→62.3 tok/s | W5 |
| w6_temperature.png | 三种温度的概率分布对比 | W6 |
| w6_top_p.png | top-p 截断示意 | W6 |
| w7_engine_arch.png | 引擎架构图 | W7 |
| w8_module_map.png | minivllm ↔ vLLM v1 模块对照 | W8 |
| w8_benchmark.png | 结业实测：minivllm vs vLLM | W8 |

- **每张图必须至少被一章引用一次**（按上表归属周引用即可）。
- 允许 ASCII 的唯一场景：**与坐标无关的纯文本示意**（如目录树、代码注释、队列状态）。
- 若某章确实需要新图：可以**新增** `figures/gen_wXdY_*.py` 脚本（模仿现有脚本：
  `sys.path.insert` 开头、用 `figures._common` 的助手、中文字体已配置），
  运行生成 PNG 后引用；**生图脚本必须保留在仓库里**，新脚本务必实际运行成功再交付。

## 5. 章节文件命名（必须与 SUMMARY.md 完全一致）

`SUMMARY.md` 已经定死了每章的文件名（含中文），**一个字都不能差**。
每周要写的文件清单见第 7 节。

## 6. 代码 API 速查（写作时以这里为准，不许凭印象编 API）

> 完整实现见 `reference/`。这里只列读者要填/要用的公开接口。

### tensor_ops.py（W1）
`make_tensor(data, dtype)` / `zeros(shape)` / `ones(shape)` / `select_row(t,i)` /
`select_column(t,j)` / `row_sums(t)` / `column_means(t)` / `matmul(a,b)` /
`to_device(t, device)` / `last_token_logits(logits)`（取 `(1,L,V)` 最后位置 → `(V,)`）/
`best_device()`（已给出）

### generate.py（W1/W3）
`generate_naive(model, input_ids, max_new_tokens, eos_token_id=None)` —— 每步整段重算；
`generate_with_cache(model, input_ids, max_new_tokens, eos_token_id=None)` —— prefill+decode（W3）；
`count_params(model)`（已给出）。input_ids 形状 `(1, L)`。

### model/attention.py（W2）
`make_causal_mask(q_len, kv_len=None)` → `(q_len, kv_len)` BoolTensor，True=允许看；
`scaled_dot_product_attention(q, k, v, mask=None)` → `(out, weights)`，
q/k/v 形状 `(B, H, L, D)`。

### model/transformer.py（W2/W3/W4）
`MiniConfig(vocab_size=128, hidden_size=64, num_layers=2, num_heads=4, max_seq_len=128, mlp_ratio=4)`；
`MiniTransformer(config)`：
- `forward(input_ids)` → logits `(B, L, V)`（W2）
- `forward(input_ids, past_kv=None, use_cache=True)` → `(logits, past_kv)`（W3）
- `prefill(input_ids(L,), kv_cache, slot_mapping, block_table)` → 最后位置 logits `(V,)`（W4）
- `decode_batch(token_ids(B,), positions(B,), kv_cache, slot_mapping, block_tables)` → `(B, V)`（W4/5）

### cache/kv_cache.py（W3）
`NaiveKVCache(num_layers)`：`append_and_get(layer_idx, k, v)` / `seq_len(layer_idx=0)` / `reset()`；
`kv_cache_memory_bytes(num_layers, num_heads, head_dim, seq_len, dtype_bytes=2)`。

### cache/block_pool.py + block_table.py + paged_cache.py（W4）
`BlockPool(num_blocks, block_size)`：`allocate()` / `free(bid)` / `num_free_blocks()`（已给出）/
`can_fit(n)`（已给出）；`OutOfBlocksError`。
`BlockTable(pool)`：`append_slot()` → 物理槽位号 / `physical_slots()` / `needed_blocks(n)`（已给出）/
`free()`（已给出）；属性 `block_ids`、`num_tokens`。
物理槽位 = 块号 × 块大小 + 块内偏移。
`PagedKVCache(num_layers, num_heads, head_dim, num_blocks, block_size, device, dtype)`：
`write(layer_idx, slot_mapping, k, v)`（k/v 形状 `(n, H, D)`）/
`gather(layer_idx, slots)` → `(L, H, D)` / `gather_padded(layer_idx, slot_lists)`（已给出）→
`(k, v, mask)`，k/v 形状 `(B, Lmax, H, D)`。

### scheduler/（W5）
`RequestStatus`（WAITING/RUNNING/FINISHED）；`Request(request_id, prompt_token_ids, sampling_params)`：
属性 `num_prompt_tokens/num_output_tokens/num_tokens`、`last_token_id()`、`is_finished()`、
`output_token_ids`、`block_table`、`finish_reason`。
`Scheduler(block_pool, max_num_seqs=8)`：`add_request(req)`（已给出）/ `has_unfinished()`（已给出）/
`schedule()` → SchedulerOutput（`.prefill` 新准入、`.decode` 在跑的）/
`update_after_step(sampled_tokens, eos_token_ids=())` → 本步结束的请求列表。
**保守准入**：块池要装得下"提示词+全部待生成"才放行（记账 `_reserved_blocks`）。
`continuous_batch_generate(model, prompts, max_new_tokens=16, num_blocks=64, block_size=4, max_num_seqs=8, eos_token_ids=())`
→ 与 prompts 同序的 `list[list[int]]`（贪心）。

### sampling/sampler.py（W6）
`SamplingParams(temperature=0.0, top_k=-1, top_p=1.0, max_new_tokens=16, stop_token_ids=(), seed=None)`
（temperature=0 即贪心）；
`greedy_sample(logits)` / `apply_temperature(logits, T)` / `top_k_filter(logits, k)` /
`top_p_filter(logits, p)` / `sample_token(logits, params=None, generator=None)`；
`Sampler()`：`sample_one(logits, request_id, params)`（已给出）/ `drop_request(rid)`（已给出）。

### model/qwen3.py（W6）
`Qwen3Config(dict)`；`RMSNorm(dim, eps)`；`RotaryEmbedding(head_dim, base)` → `(cos, sin)`；
`rotate_half(x)` / `apply_rotary_pos_emb(t, cos, sin)`；
`Qwen3Attention`（GQA：q 16 头、k/v 8 头、QK-Norm）；`Qwen3MLP`（SwiGLU）；
`Qwen3DecoderLayer`；`Qwen3ForCausalLM`：`forward(input_ids(1,L))` → `(1,L,V)`（朴素对拍用）、
`prefill/decode_batch`（引擎接口，同 MiniTransformer 签名）、
`from_pretrained(model_path, device, dtype)`（已给出）。
权重与 HF 完全同名，对拍测试见 `tests/test_w6_qwen3.py`。

### engine/（W7）
`LLMEngine(model, tokenizer, num_blocks=256, block_size=16, max_num_seqs=8, eos_token_id=None)`：
`add_request(prompt_token_ids, sampling_params)` → rid（已给出）/ `has_unfinished()`（已给出）/
`step()` → 本步完成的 rid 列表 / `generate(prompts_token_ids, sampling_params)` → `list[RequestOutput]` /
`stream(prompt_token_ids, sampling_params)` → 迭代器，逐片吐文本。
`RequestOutput`：`request_id / output_token_ids / text / finish_reason`。
`LLM(model_name="Qwen/Qwen3-0.6B", max_model_len=2048, device=None, dtype=None, num_blocks=None, block_size=16, max_num_seqs=8)`：
`generate(prompts, sampling_params)`（prompts 是字符串列表）/ `stream(prompt, sampling_params)`。

### bench/metrics.py（W8）
`BenchResult(num_prompts, num_output_tokens, total_time, ttft=None)`，
属性 `tokens_per_second`；`benchmark_throughput(generate_fn, prompts, sampling_params=None)`；
`measure_ttft(stream_fn)` → `(ttft秒, 完整文本)`；`format_report(result)`（已给出）。

### 实测数据（figures/data/bench.json，本机 Mac + Qwen3-0.6B 实测）

| 指标 | 数值 |
|---|---|
| naive_hf_tps（无 cache 整段重算） | 39.3 tok/s |
| cached_hf_tps（HF 带 cache） | 57.1 tok/s |
| minivllm_b1_tps（我们的引擎，单请求） | 19.6 tok/s |
| minivllm_b8_tps（我们的引擎，8 请求） | 62.3 tok/s |
| minivllm_ttft_ms | 62 ms |
| vllm_b1_tps（真实 vLLM，单请求） | 145.8 tok/s |
| vllm_b8_tps（真实 vLLM，8 请求） | 143.3 tok/s |

W3 小模型实测：KV cache 快 3.0 倍（4 层 hidden=256 模型，CPU）。

### 真实 vLLM 环境（第 1/8 章要用）

- **pip 安装进同一个 `.venv`，版本锁定 `vllm==0.27.1`**（不用 clone 源码仓库）：
  - macOS（Apple Silicon，需 Python 3.12）：
    `.venv/bin/pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl" "vllm-metal @ https://github.com/vllm-project/vllm-metal/releases/download/v0.3.0.dev20260819070634/vllm_metal-0.3.0.dev20260819070634-cp312-cp312-macosx_11_0_arm64.whl"`
    （注意：PyPI 上的 vllm-metal 0.1.0 与 vLLM 0.27 不兼容，必须用 GitHub 上的 0.3.0.dev 构建）
  - Linux：`.venv/bin/pip install vllm==0.27.1`
- 源码就在 venv 里：`.venv/lib/python3.12/site-packages/vllm/`；
  教程引用真实源码时直接写包内相对路径（如 `vllm/v1/core/sched/scheduler.py`）。
- 本机是 macOS 14.1，预编译 Metal kernel 需要 macOS 15+，所以必须加环境变量
  `VLLM_METAL_USE_PAGED_ATTENTION=0`（退回 MLX 自带注意力）。
- vLLM 会 spawn 独立的 EngineCore 子进程，**必须从带 `if __name__ == "__main__":`
  守卫的 .py 脚本文件运行**，不能用 `python - <<EOF` 或 `python -c`。
- 模型走本地 HF 缓存：`HF_HUB_OFFLINE=1`。

## 7. 每周任务分派

> 每周写一个目录 `learning-guide/weekN/`：`README.md`（本周导读）+ 下表所列章节。
> README.md 要求：本周定位、目录表（链接到各章）、"学完你会得到什么"、常用命令速查。
> 每章末尾的"动手练习"指向当周测试文件：`pytest tests/test_wN.py`（或 `-k` 具体用例）。

### Week 1：见证与上手——一个词一个词是怎么蹦出来的（⭐ 零基础可入）

| 文件 | 主题与要点 |
|---|---|
| 00-总览-vLLM为什么快.md | 从"打字机效果"聊起；推理引擎管什么；课程地图（8 周表）；怎么用本教程（战场/参考答案/测试三件套）；承诺"不用懂 CUDA、不用懂操作系统" |
| 01-环境搭建-装好工具先见证奇迹.md | python 版本检查（建议 3.12）、`.venv` + `pip install -r requirements.txt`；跑 `IMPL=reference pytest tests` 看全绿（见证终点）；跑 `pytest tests` 看红；pip 安装真实 vLLM（锁定 v0.27.1，macOS 用 GitHub release wheel + vllm-metal，注意 `VLLM_METAL_USE_PAGED_ATTENTION=0`、必须脚本文件运行）生成第一段话；源码约定框（源码在 venv 的 site-packages 里，定位命令）；图 w1_benchmark.png |
| 02-PyTorch热身-张量就是数表.md | tensor 创建/索引/矩阵乘/device（mps）；逐个实现 tensor_ops.py；跑 `pytest tests/test_w1.py -k "not generate"` |
| 03-自回归-一个词一个词蹦出来.md | 下一个词预测；手写 generate_naive（先对 StubHFModel 跑通，再换真实 HF Qwen3）；图 w1_autoregressive.png；跑 `pytest tests/test_w1.py` |
| 04-AI联系-推理引擎是干什么的.md | 训练 vs 推理；吞吐/延迟；vLLM 在生态里的位置；预告 Week 2 |

### Week 2：手写 mini Transformer——引擎的心脏（⭐⭐）

| 文件 | 主题与要点 |
|---|---|
| 01-注意力-每个词都回头看看.md | 用"回头看前文"建直觉；Q/K/V 各管什么（查询/标签/内容）；scaled_dot_product_attention 四行数学；softmax 一句话公式+直觉 |
| 02-因果掩码-不许偷看未来.md | make_causal_mask；q_len≠kv_len 的情形（为 W3 埋点）；手算 2x2 例子；跑 `pytest tests/test_w2.py -k "mask or sdpa"` |
| 03-多头与残差-把零件拼成一层.md | 多头=多双眼睛；_split_heads/_merge_heads（已给出，讲透形状）；LayerNorm/残差一句话直觉；MLP |
| 04-组装-MiniTransformer诞生.md | embedding（词→坐标）；堆叠 N 层；lm_head；跑 `pytest tests/test_w2.py`；图 w2_transformer_map.png + w2_attention_heatmap.png（我们自己模型的真实权重） |
| 05-AI联系-这就是vLLM里的model.md | 对照 `vllm/model_executor/models/`；预告 Week 3 的重复计算问题 |

### Week 3：KV cache——别重复算已经算过的（⭐⭐）

| 文件 | 主题与要点 |
|---|---|
| 01-浪费在哪-每步重算整段话.md | 逐步数计算量；图 w3_recompute_waste.png |
| 02-KVcache-把中间结果存下来.md | K/V 是"每个词的档案"；NaiveKVCache.append_and_get；concat 的代价一句话带过 |
| 03-prefill与decode-一次吃饱与一口一口吃.md | generate_with_cache；两个阶段命名；forward(use_cache=True)；跑 `pytest tests/test_w3.py` |
| 04-实测加速与内存账.md | 图 w3_speedup.png（3.0 倍）；kv_cache_memory_bytes 手算 Qwen3-0.6B 每 token 112KB；图 w3_memory_bill.png |
| 05-AI联系-KVcache是推理的头号内存.md | 内存=并发请求数的瓶颈；GQA 预告（W6）；PagedAttention 预告（W4） |

### Week 4：分页——向操作系统借智慧（⭐⭐⭐ 灵魂周之一）

| 文件 | 主题与要点 |
|---|---|
| 01-内存碎片-储物柜的智慧.md | 连续分配的洞；固定大小块+页表的分页思想（储物柜类比）；图 w4_fragmentation.png |
| 02-BlockPool与BlockTable-页表本尊.md | 实现 allocate/free/append_slot/physical_slots；物理槽位公式；图 w4_block_table.png；跑 `pytest tests/test_w4.py -k "pool or block_table"` |
| 03-PagedKVCache-写进去读回来.md | write/gather；gather_padded（已给出，讲读）；跑 `pytest tests/test_w4.py -k paged` |
| 04-模型的分页接口-prefill与decode_batch.md | MiniTransformer 长出引擎接口；slot_mapping 谁负责（调度层）；补 0 对齐+掩码的批量 decode |
| 05-铁证-分页和不分页一字不差.md | test_paged_equals_naive 对拍实验逐行讲；跑 `pytest tests/test_w4.py` |
| 06-AI联系-PagedAttention到底省了什么.md | 内存利用率→更多并发；真实 vLLM 的 kernel 一句话（我们 gather 成连续再算，真实直接读块） |

### Week 5：调度器——continuous batching（⭐⭐⭐ 灵魂周之二）

| 文件 | 主题与要点 |
|---|---|
| 01-静态批处理-陪跑的浪费.md | 甘特图讲故事；图 w5_gantt.png |
| 02-Request与两个队列-waiting与running.md | 请求状态机；finish_reason（length/stop） |
| 03-Scheduler-每步点名.md | schedule/update_after_step；保守准入与 `_reserved_blocks` 记账；跑 `pytest tests/test_w5.py -k scheduler` |
| 04-组装流水线-continuous_batch_generate.md | 全链路串一遍（读者实现这个函数）；跑 `pytest tests/test_w5.py` |
| 05-实测-批处理把吞吐抬高3倍.md | 图 w5_throughput.png（19.6→62.3）；为什么 b8 提升明显 |
| 06-AI联系-真实vLLM调度器还多了什么.md | token budget、抢占、chunked prefill（各一段，指向 material/vllm-源码对照.md） |

### Week 6：采样与真实权重——让引擎说人话（⭐⭐）

| 文件 | 主题与要点 |
|---|---|
| 01-从logits到词-greedy与temperature.md | logits→softmax→概率；温度缩放；图 w6_temperature.png；跑 `pytest tests/test_w6_sampler.py -k "greedy or temperature"` |
| 02-topk与topp-给随机性画个圈.md | top_k_filter/top_p_filter 逐个实现；图 w6_top_p.png；跑 `-k "top"` |
| 03-Sampler-每个请求一套参数.md | SamplingParams；seed 复现；sample_token 流水线；跑 `pytest tests/test_w6_sampler.py` |
| 04-Qwen3的四处升级-RMSNorm与SwiGLU.md | 从 MiniTransformer 出发的四处升级（先讲这两处）；实现 RMSNorm/Qwen3MLP |
| 05-RoPE与GQA-位置编码与省内存.md | rotate_half/apply_rotary_pos_emb；GQA 的 repeat_interleave；QK-Norm 一句话 |
| 06-加载真实权重-和HuggingFace对拍.md | from_pretrained（已给出，讲流程）；实现 attention/层/模型三条路径；对拍测试逐行讲；跑 `pytest tests/test_w6_qwen3.py -m slow` |
| 07-AI联系-模型文件里都装了什么.md | safetensors/config.json/tokenizer 三件套；tie_word_embeddings |

### Week 7：引擎化——从脚本到服务（⭐⭐）

| 文件 | 主题与要点 |
|---|---|
| 01-从函数到引擎-为什么要LLMEngine.md | W5 函数的局限（一次进一批）；长跑引擎的需求；图 w7_engine_arch.png |
| 02-step-引擎的心跳.md | step() 逐行讲（点名→prefill→decode→采样→登记）；实现 step |
| 03-generate与stream-两种用法.md | generate 循环；stream 的增量 decode（`text[len(sent):]`）；实现二者 |
| 04-LLM门面-一行起引擎.md | LLM 类（已给出 __init__，讲每一行）；实现 generate/stream 包装 |
| 05-实战-多人同时聊天.md | 用 LLM 批量+流式生成真实 Qwen3；跑 `pytest tests/test_w7.py`（含 slow） |
| 06-AI联系-离vllmLLM还差什么.md | API server、AsyncLLM、多进程；一句话指向 vllm/entrypoints/ |

### Week 8：对标真实 vLLM——读懂源码（⭐⭐⭐ 结业周）

| 文件 | 主题与要点 |
|---|---|
| 01-模块对照-你已经认识它们了.md | 图 w8_module_map.png；material/vllm-源码对照.md 总览 |
| 02-带读vLLM调度器源码.md | 带读 vllm/v1/core/sched/scheduler.py 的 schedule() 前 100 行 + request.py + block_pool.py 的 get_new_blocks |
| 03-高级特性导览-prefixcaching与chunkedprefill.md | 各一节：解决什么、大概怎么做、在真实源码的哪里 |
| 04-性能测量-TTFT与吞吐.md | 实现 bench/metrics.py；两个指标的意义；跑 `pytest tests/test_w8.py` |
| 05-结业实测-我们追到了几成功力.md | 图 w8_benchmark.png；差距在哪（kernel、varlen、chunked prefill、前缀缓存） |
| 06-大结业-完整旅程回顾与下一步.md | 8 周地图回顾；下一步（投机采样、量化、多卡）；致谢 |

## 8. 交付要求（子 agent 写完一周后要自查）

1. 文件名与 SUMMARY.md 一字不差；每章有 `> 本章你将：` 导读、固定盒子、动手练习、参考答案；
2. 本周归属的图全部被引用（第 4 节的表）；不新增坐标类 ASCII 图；
3. 教程里出现的代码/API 与 reference/ 一致；引用真实 vLLM 时给出仓库内路径；
4. 引用实测数字时与 figures/data/bench.json 一致（不要编数字）；
5. 交付说明：列出写了哪些文件、引用了哪些图、自查发现的问题。
