"""Week 1：PyTorch 上手——张量操作热身。

这些函数是后面所有章节的"螺丝刀"。它们故意做得很小，
目的是让你把 torch 的基本操作（创建、索引、矩阵乘、搬运到 GPU）练熟。

约定：
- 本教程里所有"张量"都是 torch.Tensor；
- device 用字符串表示，比如 "cpu"、"mps"（Mac 的 GPU）。
"""

import torch


def make_tensor(data, dtype=torch.float32):
    """把嵌套列表变成 torch 张量。

    >>> t = make_tensor([[1, 2], [3, 4]])
    >>> t.shape
    torch.Size([2, 2])
    """
    return torch.tensor(data, dtype=dtype)


def zeros(shape):
    """造一个全 0 张量。shape 可以是元组，例如 (2, 3)。"""
    return torch.zeros(shape)


def ones(shape):
    """造一个全 1 张量。shape 可以是元组，例如 (2, 3)。"""
    return torch.ones(shape)


def select_row(t, i):
    """取第 i 行（下标从 0 开始），返回一个一维张量。"""
    return t[i]


def select_column(t, j):
    """取第 j 列，返回一个一维张量。"""
    return t[:, j]


def row_sums(t):
    """对每一行求和，返回一维张量（长度为行数）。"""
    return t.sum(dim=1)


def column_means(t):
    """对每一列求平均，返回一维张量（长度为列数）。"""
    return t.mean(dim=0)


def matmul(a, b):
    """矩阵乘法：a 的形状 (m, k)，b 的形状 (k, n)，返回 (m, n)。"""
    return a @ b


def to_device(t, device):
    """把张量搬到指定设备（"cpu" 或 "mps"），返回搬运后的张量。"""
    return t.to(device)


def best_device():
    """返回本机最快的可用设备名：有 MPS（Mac GPU）用 "mps"，否则 "cpu"。"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_token_logits(logits):
    """从模型输出里取"最后一个位置"的 logits。

    模型一次前向会给序列里每个位置都算一份"下一个词"的预测，
    形状是 (batch, seq_len, vocab_size)。我们生成时只关心最后一个位置。

    返回形状 (vocab_size,) 的一维张量（假设 batch=1）。
    """
    return logits[0, -1, :]
