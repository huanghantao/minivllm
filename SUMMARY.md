# Summary

[课程介绍](README.md)

# Week 1：见证与上手——一个词一个词是怎么蹦出来的

- [本周导读](learning-guide/week1/README.md)
- [第 0 章：总览——vLLM 为什么快](learning-guide/week1/00-总览-vLLM为什么快.md)
- [第 1 章：环境搭建——装好工具，先见证奇迹](learning-guide/week1/01-环境搭建-装好工具先见证奇迹.md)
- [第 2 章：PyTorch 热身——张量就是数表](learning-guide/week1/02-PyTorch热身-张量就是数表.md)
- [第 3 章：自回归——一个词一个词蹦出来](learning-guide/week1/03-自回归-一个词一个词蹦出来.md)
- [第 4 章：AI 联系——推理引擎是干什么的](learning-guide/week1/04-AI联系-推理引擎是干什么的.md)

# Week 2：手写 mini Transformer——引擎的心脏

- [本周导读](learning-guide/week2/README.md)
- [第 1 章：注意力——每个词都回头看看](learning-guide/week2/01-注意力-每个词都回头看看.md)
- [第 2 章：因果掩码——不许偷看未来](learning-guide/week2/02-因果掩码-不许偷看未来.md)
- [第 3 章：多头与残差——把零件拼成一层](learning-guide/week2/03-多头与残差-把零件拼成一层.md)
- [第 4 章：组装——MiniTransformer 诞生](learning-guide/week2/04-组装-MiniTransformer诞生.md)
- [第 5 章：AI 联系——这就是 vLLM 里的 model](learning-guide/week2/05-AI联系-这就是vLLM里的model.md)

# Week 3：KV cache——别重复算已经算过的

- [本周导读](learning-guide/week3/README.md)
- [第 1 章：浪费在哪——每步重算整段话](learning-guide/week3/01-浪费在哪-每步重算整段话.md)
- [第 2 章：KV cache——把中间结果存下来](learning-guide/week3/02-KVcache-把中间结果存下来.md)
- [第 3 章：prefill 与 decode——一次吃饱与一口一口吃](learning-guide/week3/03-prefill与decode-一次吃饱与一口一口吃.md)
- [第 4 章：实测加速与内存账](learning-guide/week3/04-实测加速与内存账.md)
- [第 5 章：AI 联系——KV cache 是推理的头号内存](learning-guide/week3/05-AI联系-KVcache是推理的头号内存.md)

# Week 4：分页——向操作系统借智慧

- [本周导读](learning-guide/week4/README.md)
- [第 1 章：内存碎片——储物柜的智慧](learning-guide/week4/01-内存碎片-储物柜的智慧.md)
- [第 2 章：BlockPool 与 BlockTable——页表本尊](learning-guide/week4/02-BlockPool与BlockTable-页表本尊.md)
- [第 3 章：PagedKVCache——写进去，读回来](learning-guide/week4/03-PagedKVCache-写进去读回来.md)
- [第 4 章：模型的分页接口——prefill 与 decode_batch](learning-guide/week4/04-模型的分页接口-prefill与decode_batch.md)
- [第 5 章：铁证——分页和不分页一字不差](learning-guide/week4/05-铁证-分页和不分页一字不差.md)
- [第 6 章：AI 联系——PagedAttention 到底省了什么](learning-guide/week4/06-AI联系-PagedAttention到底省了什么.md)

# Week 5：调度器——continuous batching

- [本周导读](learning-guide/week5/README.md)
- [第 1 章：静态批处理——陪跑的浪费](learning-guide/week5/01-静态批处理-陪跑的浪费.md)
- [第 2 章：Request 与两个队列——waiting 与 running](learning-guide/week5/02-Request与两个队列-waiting与running.md)
- [第 3 章：Scheduler——每步点名](learning-guide/week5/03-Scheduler-每步点名.md)
- [第 4 章：组装流水线——continuous_batch_generate](learning-guide/week5/04-组装流水线-continuous_batch_generate.md)
- [第 5 章：实测——批处理把吞吐抬高 3 倍](learning-guide/week5/05-实测-批处理把吞吐抬高3倍.md)
- [第 6 章：AI 联系——真实 vLLM 调度器还多了什么](learning-guide/week5/06-AI联系-真实vLLM调度器还多了什么.md)

# Week 6：采样与真实权重——让引擎说人话

- [本周导读](learning-guide/week6/README.md)
- [第 1 章：从 logits 到词——greedy 与 temperature](learning-guide/week6/01-从logits到词-greedy与temperature.md)
- [第 2 章：top-k 与 top-p——给随机性画个圈](learning-guide/week6/02-topk与topp-给随机性画个圈.md)
- [第 3 章：Sampler——每个请求一套参数](learning-guide/week6/03-Sampler-每个请求一套参数.md)
- [第 4 章：Qwen3 的四处升级——RMSNorm 与 SwiGLU](learning-guide/week6/04-Qwen3的四处升级-RMSNorm与SwiGLU.md)
- [第 5 章：RoPE 与 GQA——位置编码与省内存](learning-guide/week6/05-RoPE与GQA-位置编码与省内存.md)
- [第 6 章：加载真实权重——和 HuggingFace 对拍](learning-guide/week6/06-加载真实权重-和HuggingFace对拍.md)
- [第 7 章：AI 联系——模型文件里都装了什么](learning-guide/week6/07-AI联系-模型文件里都装了什么.md)

# Week 7：引擎化——从脚本到服务

- [本周导读](learning-guide/week7/README.md)
- [第 1 章：从函数到引擎——为什么要 LLMEngine](learning-guide/week7/01-从函数到引擎-为什么要LLMEngine.md)
- [第 2 章：step——引擎的心跳](learning-guide/week7/02-step-引擎的心跳.md)
- [第 3 章：generate 与 stream——两种用法](learning-guide/week7/03-generate与stream-两种用法.md)
- [第 4 章：LLM 门面——一行起引擎](learning-guide/week7/04-LLM门面-一行起引擎.md)
- [第 5 章：实战——多人同时聊天](learning-guide/week7/05-实战-多人同时聊天.md)
- [第 6 章：AI 联系——离 vllm.LLM 还差什么](learning-guide/week7/06-AI联系-离vllmLLM还差什么.md)

# Week 8：对标真实 vLLM——读懂源码

- [本周导读](learning-guide/week8/README.md)
- [第 1 章：模块对照——你已经认识它们了](learning-guide/week8/01-模块对照-你已经认识它们了.md)
- [第 2 章：带读 vLLM 调度器源码](learning-guide/week8/02-带读vLLM调度器源码.md)
- [第 3 章：高级特性导览——prefix caching 与 chunked prefill](learning-guide/week8/03-高级特性导览-prefixcaching与chunkedprefill.md)
- [第 4 章：性能测量——TTFT 与吞吐](learning-guide/week8/04-性能测量-TTFT与吞吐.md)
- [第 5 章：结业实测——我们追到了几成功力](learning-guide/week8/05-结业实测-我们追到了几成功力.md)
- [第 6 章：大结业——完整旅程回顾与下一步](learning-guide/week8/06-大结业-完整旅程回顾与下一步.md)
