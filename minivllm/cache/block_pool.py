"""Week 4：BlockPool——KV cache 的"储物柜管理员"。

思想借自操作系统的内存分页：
- 把整块 KV cache 内存切成固定大小的"块"（block）；
- 请求来了按需领块，用完了归还；
- 块不必连续，碎不了。

BlockPool 只负责一件事：记录哪些块闲着，发块、收块。
"""

from collections import deque


class OutOfBlocksError(RuntimeError):
    """池子里没有空闲块了。"""


class BlockPool:
    """管理 num_blocks 个固定大小的块。

    块的编号是 0..num_blocks-1；每个块能装 block_size 个 token 的 K/V。
    """

    def __init__(self, num_blocks, block_size):
        assert num_blocks > 0 and block_size > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        # 空闲块队列。用 deque 让取块/还块都是 O(1)。
        self._free = deque(range(num_blocks))

    def allocate(self):
        """领一个空闲块，返回块号。池空则抛 OutOfBlocksError。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # BlockPool.allocate

    def free(self, block_id):
        """归还一个块。"""
        raise NotImplementedError("TODO: 看 learning-guide 对应章节，把这个函数填出来")  # BlockPool.free

    def num_free_blocks(self):
        """还剩多少空闲块。"""
        return len(self._free)

    def can_fit(self, num_tokens):
        """池子还装得下 num_tokens 个 token 吗？"""
        blocks_needed = -(-num_tokens // self.block_size)  # 向上取整
        return blocks_needed <= self.num_free_blocks()
