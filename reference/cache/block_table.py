"""Week 4：BlockTable——一个请求的"页表"。

操作系统里，页表记录"程序眼里的第几页 → 内存里的第几页框"。
这里一样：块表记录"请求的第几段 token → KV cache 里的第几块"。

请求眼里自己的 token 是连续的（0, 1, 2, ...），
但物理上它们分散在一个个块里。块表就是做这道翻译。
"""


class BlockTable:
    """一个请求的 逻辑位置 → 物理槽位 对照表。

    物理槽位 = 块号 × 块大小 + 块内偏移。
    """

    def __init__(self, pool):
        self.pool = pool
        self.block_ids = []  # 这个请求占用的块号，按逻辑顺序排
        self.num_tokens = 0  # 已经占用了多少个槽位

    def append_slot(self):
        """为下一个 token 占一个槽位，返回它的物理槽位号。

        如果当前最后一块满了（或还没有块），先向池子领一个新块。
        """
        block_size = self.pool.block_size
        if self.num_tokens % block_size == 0:
            # 需要新开一个块（第一个 token 也会走到这里）
            self.block_ids.append(self.pool.allocate())
        block_id = self.block_ids[-1]
        offset = self.num_tokens % block_size
        self.num_tokens += 1
        return block_id * block_size + offset

    def physical_slots(self):
        """返回全部 token 的物理槽位，按逻辑顺序（一个 list[int]）。"""
        block_size = self.pool.block_size
        slots = []
        for i in range(self.num_tokens):
            block_id = self.block_ids[i // block_size]
            slots.append(block_id * block_size + i % block_size)
        return slots

    def needed_blocks(self, extra_tokens):
        """再装 extra_tokens 个 token，还需要几个新块？"""
        block_size = self.pool.block_size
        total = self.num_tokens + extra_tokens
        return -(-total // block_size) - len(self.block_ids)

    def free(self):
        """把占用的块全部还给池子（请求结束时调用）。"""
        for block_id in self.block_ids:
            self.pool.free(block_id)
        self.block_ids = []
        self.num_tokens = 0
