"""Week 1：张量热身 + 朴素生成循环。"""

import torch

from tests._impl import load

tensor_ops = load("tensor_ops")
generate_mod = load("generate")


# ---------- tensor_ops ----------

def test_make_tensor():
    t = tensor_ops.make_tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.dtype == torch.float32
    assert torch.allclose(t, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_zeros_ones():
    assert tensor_ops.zeros((2, 3)).shape == (2, 3)
    assert tensor_ops.zeros((2, 3)).sum().item() == 0
    assert tensor_ops.ones((2, 3)).sum().item() == 6


def test_select_row_column():
    t = tensor_ops.make_tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.equal(tensor_ops.select_row(t, 1), torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(tensor_ops.select_column(t, 0), torch.tensor([1.0, 4.0]))


def test_row_sums_column_means():
    t = tensor_ops.make_tensor([[1, 2], [3, 4]])
    assert torch.equal(tensor_ops.row_sums(t), torch.tensor([3.0, 7.0]))
    assert torch.equal(tensor_ops.column_means(t), torch.tensor([2.0, 3.0]))


def test_matmul():
    a = tensor_ops.make_tensor([[1, 2], [3, 4]])
    b = tensor_ops.make_tensor([[5, 6], [7, 8]])
    assert torch.equal(tensor_ops.matmul(a, b), torch.tensor([[19.0, 22.0], [43.0, 50.0]]))


def test_to_device_roundtrip():
    t = tensor_ops.make_tensor([1, 2, 3])
    assert tensor_ops.to_device(t, "cpu").device.type == "cpu"


def test_best_device_is_valid():
    assert tensor_ops.best_device() in ("cpu", "mps")


def test_last_token_logits():
    logits = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    out = tensor_ops.last_token_logits(logits)
    assert out.shape == (4,)
    assert torch.equal(out, torch.tensor([8.0, 9.0, 10.0, 11.0]))


# ---------- generate_naive ----------

class StubHFModel:
    """假 HF 模型：查表决定下一个词。table: {上一个词: 下一个词}"""

    class Out:
        def __init__(self, logits):
            self.logits = logits

    def __init__(self, table, vocab_size=8):
        self.table = table
        self.vocab_size = vocab_size

    def __call__(self, ids):
        last = int(ids[0, -1].item())
        nxt = self.table[last]
        logits = torch.zeros(1, ids.shape[1], self.vocab_size)
        logits[0, -1, nxt] = 10.0  # argmax 必中
        return self.Out(logits)


def test_generate_naive_follows_chain():
    model = StubHFModel({1: 2, 2: 3, 3: 7})
    ids = torch.tensor([[1]])
    out = generate_mod.generate_naive(model, ids, max_new_tokens=10, eos_token_id=7)
    assert out[0].tolist() == [1, 2, 3, 7]


def test_generate_naive_respects_max_tokens():
    model = StubHFModel({1: 2, 2: 3, 3: 4, 4: 5, 5: 6})
    ids = torch.tensor([[1]])
    out = generate_mod.generate_naive(model, ids, max_new_tokens=3)
    assert out[0].tolist() == [1, 2, 3, 4]  # 恰好 3 个新词


def test_count_params(mini_model):
    n = generate_mod.count_params(mini_model)
    assert n == sum(p.numel() for p in mini_model.parameters())
    assert n > 0
