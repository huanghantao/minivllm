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
        if len(self.running) >= self.max_num_seqs:
            return False
        needed = self._blocks_needed(request, self.block_pool.block_size)
        available = self.block_pool.num_free_blocks() - self._reserved_blocks
        return needed <= available

    def schedule(self):
        """点名一次，返回 SchedulerOutput。

        - 队首请求若能准入（有位子、块够），就准入，直到队首进不来为止；
        - 新准入的本步 prefill；所有在跑的本步 decode。
          （刚准入的只 prefill，它的第一次 decode 在下一步。）
        """
        out = SchedulerOutput()
        while self.waiting and self._can_admit(self.waiting[0]):
            req = self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            req.block_table = BlockTable(self.block_pool)
            req.reserved_blocks = self._blocks_needed(req, self.block_pool.block_size)
            self._reserved_blocks += req.reserved_blocks
            self.running.append(req)
            out.prefill.append(req)
        out.decode = [r for r in self.running if r not in out.prefill]
        return out

    def update_after_step(self, sampled_tokens, eos_token_ids=()):
        """登记本步采出的 token，把做完的请求请下车。

        sampled_tokens: {request_id: token_id}，本步每个跑过的请求的新词。
        eos_token_ids: 结束符集合，撞上即以 "stop" 结束。

        返回本步刚结束的请求列表。
        """
        newly_finished = []
        for req in list(self.running):
            if req.request_id not in sampled_tokens:
                continue
            token = sampled_tokens[req.request_id]
            req.output_token_ids.append(token)

            hit_length = (
                req.num_output_tokens >= req.sampling_params.max_new_tokens
            )
            hit_stop = token in eos_token_ids or token in req.sampling_params.stop_token_ids
            if hit_length or hit_stop:
                req.status = RequestStatus.FINISHED
                req.finish_reason = "length" if hit_length else "stop"
                req.block_table.free()  # 块还回池子，别人才能进来
                self._reserved_blocks -= req.reserved_blocks
                self.running.remove(req)
                self.finished.append(req)
                newly_finished.append(req)
        return newly_finished
