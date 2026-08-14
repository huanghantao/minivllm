"""Week 5：Scheduler——continuous batching 的调度员。

静态批处理的问题：一锅请求必须同时进站、同时出锅，
短请求做完了只能干等长请求，GPU 空转。

continuous batching（连续批处理）：
- 每一步都重新点名——谁做完了谁走，谁排到了谁上；
- GPU 上永远坐满正在生成的请求。

本调度器的简化（相对真实 vLLM）：
- 准入是保守的：只有块池装得下"提示词 + 全部待生成 token"才放行，
  这样运行中途绝不会缺块，也就不需要 vLLM 的抢占（preemption）机制；
- prefill 不切块（chunked prefill 留到 Week 8 导览）。
"""

from collections import deque

from ..cache.block_table import BlockTable
from .request import Request, RequestStatus


class SchedulerOutput:
    """一次点名的结果：这一步该谁 prefill、该谁 decode。"""

    def __init__(self):
        self.prefill = []  # 新准入的请求（本步跑 prefill + 采第一个词）
        self.decode = []   # 已经在跑的请求（本步各解一个 token）

    def is_empty(self):
        return not self.prefill and not self.decode


class Scheduler:
    """管理 waiting / running 两个队列和块池的调度员。"""

    def __init__(self, block_pool, max_num_seqs=8):
        self.block_pool = block_pool
        self.max_num_seqs = max_num_seqs
        self.waiting = deque()   # Request 队列，先到先服务
        self.running = []        # 正在生成的 Request
        self.finished = []       # 已结束的 Request
        self._reserved_blocks = 0  # 已准入请求"预订"的块数（物理块按需分配，账先记上）

    def add_request(self, request):
        """新请求进门，排到队尾。"""
        self.waiting.append(request)

    def has_unfinished(self):
        """还有没做完的请求吗？"""
        return len(self.waiting) + len(self.running) > 0

    @staticmethod
    def _blocks_needed(request, block_size):
        total = request.num_prompt_tokens + request.sampling_params.max_new_tokens
        return -(-total // block_size)  # 向上取整

    def _can_admit(self, request):
        """保守准入：同时满足"批次有空位"和"块池装得下全程"才放行。

        注意要减掉其他请求已预订的块——它们虽然还没真正领走物理块，
        但账上已经不属于后来人了。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # Scheduler._can_admit

    def schedule(self):
        """点名一次，返回 SchedulerOutput。

        - 队首请求若能准入（有位子、块够），就准入，直到队首进不来为止；
        - 新准入的本步 prefill；所有在跑的本步 decode。
          （刚准入的只 prefill，它的第一次 decode 在下一步。）
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # Scheduler.schedule

    def update_after_step(self, sampled_tokens, eos_token_ids=()):
        """登记本步采出的 token，把做完的请求请下车。

        sampled_tokens: {request_id: token_id}，本步每个跑过的请求的新词。
        eos_token_ids: 结束符集合，撞上即以 "stop" 结束。

        返回本步刚结束的请求列表。
        """
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # Scheduler.update_after_step
