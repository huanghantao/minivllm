"""Week 5：Request——引擎里流转的"工单"。

一个请求从进门到出门经历三种状态：
    WAITING（排队中）→ RUNNING（在算）→ FINISHED（做完了/被停了）

调度器（scheduler.py）每走一步就看一眼所有工单，决定这一步给谁算。
"""

from enum import Enum, auto

from ..sampling.sampler import SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Request:
    """一个生成请求。

    属性里最有故事的几个：
    - prompt_token_ids：提示词（用户给的）；
    - output_token_ids：已经生成出来的词；
    - block_table：它的 KV cache"页表"（被调度器分配后才有）；
    - finish_reason：结束原因——"length"（够长了）或 "stop"（撞见结束符）。
    """

    def __init__(self, request_id, prompt_token_ids, sampling_params=None):
        assert len(prompt_token_ids) > 0, "提示词不能为空"
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.status = RequestStatus.WAITING
        self.output_token_ids = []
        self.block_table = None
        self.reserved_blocks = 0  # 准入时向块池预订的块数（调度器记账用）
        self.finish_reason = None

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    @property
    def num_tokens(self):
        """它一共占多少个 KV 槽位（提示词 + 已生成）。"""
        return self.num_prompt_tokens + self.num_output_tokens

    def last_token_id(self):
        """下一步要喂给模型的那个 token。"""
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def is_finished(self):
        return self.status is RequestStatus.FINISHED

    def __repr__(self):
        return (
            f"Request(id={self.request_id}, {self.status.name}, "
            f"prompt={self.num_prompt_tokens}, out={self.num_output_tokens})"
        )
