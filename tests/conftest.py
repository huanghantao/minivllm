"""共享 fixture：小模型、小配置、假分词器。"""

import pytest
import torch

from tests._impl import load


@pytest.fixture
def mini_config():
    tf = load("model.transformer")
    return tf.MiniConfig(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=64
    )


@pytest.fixture
def mini_model(mini_config):
    torch.manual_seed(0)
    tf = load("model.transformer")
    model = tf.MiniTransformer(mini_config)
    model.eval()
    return model


class ToyTokenizer:
    """一个玩具分词器：按空格切词，词表是从训练语料里现建的。

    只实现引擎需要的两个方法：encode / decode。
    """

    def __init__(self, corpus):
        vocab = sorted({w for text in corpus for w in text.split()})
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.eos_token_id = None

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        # 模型可能吐出词表外的 id（随机权重嘛），用占位符兜住
        return " ".join(self.itos.get(i, f"<{i}>") for i in ids)
