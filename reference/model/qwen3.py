"""Week 6：Qwen3——给我们的引擎换一颗真的心脏。

MiniTransformer（Week 2）学会了 Transformer 的全部招式，
但它是"教学版"。真实模型 Qwen3 只差四步升级：

1. LayerNorm → RMSNorm（更省，效果相当）；
2. 学习位置嵌入 → RoPE 旋转位置编码（相对位置，外推性好）；
3. 多头注意力 MHA → 分组查询注意力 GQA（K/V 头少一些，KV cache 直接减半+）；
4. GELU 版的 MLP → SwiGLU（门控分支）。

权重从 HuggingFace 的 safetensors  checkpoint 直接读进来，
模块命名与官方完全一致，所以能对上号。
"""

import json
import math
import os

import torch
import torch.nn as nn

from .attention import make_causal_mask, scaled_dot_product_attention


class Qwen3Config:
    """从 HF 的 config.json 读进来的配置。"""

    def __init__(self, cfg: dict):
        self.vocab_size = cfg["vocab_size"]
        self.hidden_size = cfg["hidden_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = cfg["intermediate_size"]
        self.rms_norm_eps = cfg["rms_norm_eps"]
        self.rope_theta = cfg.get("rope_theta", 1000000.0)
        self.max_position_embeddings = cfg.get("max_position_embeddings", 40960)
        self.tie_word_embeddings = cfg.get("tie_word_embeddings", False)
        self.eos_token_id = cfg.get("eos_token_id")
        # 引擎接口要用的别名（与 MiniConfig 对齐）
        self.num_kv_groups = self.num_heads // self.num_kv_heads


class RMSNorm(nn.Module):
    """均方根归一化：x / sqrt(mean(x²) + eps) * weight。没有减均值、没有 bias。"""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE：把每两个维度看成一个小指针，位置越靠后转得越多。

    预存 inv_freq = theta^(-2i/d)，用的时候算 cos/sin。
    """

    def __init__(self, head_dim, base):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        """positions: (n,) 的整数位置。返回 (cos, sin)，形状 (n, head_dim)。"""
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # (n, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (n, D)，rotate_half 的配对方式
        return emb.cos(), emb.sin()


def rotate_half(x):
    """把最后维对半切开、交换并取负：[x1, x2] -> [-x2, x1]。"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(t, cos, sin):
    """给 q/k 施加旋转。t: (..., head_dim)，cos/sin 可广播到 t。"""
    return t * cos + rotate_half(t) * sin


class Qwen3Attention(nn.Module):
    """GQA 注意力：Q 有 num_heads 个头，K/V 只有 num_kv_heads 个头。

    Qwen3 还在 q/k 上各加了一个 RMSNorm（QK-Norm），稳住注意力分数。
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        E = config.hidden_size
        self.q_proj = nn.Linear(E, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(E, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(E, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, E, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def _project_qkv(self, x):
        """x: (n, E)。返回 q: (n, H, D)，k/v: (n, Hkv, D)，并完成 QK-Norm。"""
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v

    def _repeat_kv(self, t):
        """GQA 的关键：把 K/V 头复制 num_kv_groups 份，凑成和 Q 一样多。

        t: (n, Hkv, D) -> (n, H, D)
        """
        return t.repeat_interleave(self.num_kv_groups, dim=1)

    # ---- 朴素路径（Week 6 对拍用）：整段一次算完，无 cache ----

    def naive_forward(self, x, cos, sin):
        """x: (L, E)。返回 (L, E)。"""
        L = x.shape[0]
        q, k, v = self._project_qkv(x)
        q = apply_rotary_pos_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rotary_pos_emb(k, cos.unsqueeze(1), sin.unsqueeze(1))
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        # 换成 (1, H, L, D) 做注意力
        q4 = q.permute(1, 0, 2).unsqueeze(0)
        k4 = k.permute(1, 0, 2).unsqueeze(0)
        v4 = v.permute(1, 0, 2).unsqueeze(0)
        mask = make_causal_mask(L, L).to(x.device)
        out, _ = scaled_dot_product_attention(q4, k4, v4, mask)
        out = out.squeeze(0).permute(1, 0, 2).reshape(L, -1)
        return self.o_proj(out)

    # ---- 分页引擎接口 ----

    def prefill(self, x, cos, sin, kv_cache, slot_mapping, block_table):
        """一个请求的整段提示词。x: (L, E)。返回 (L, E)。"""
        L = x.shape[0]
        q, k, v = self._project_qkv(x)
        q = apply_rotary_pos_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k = apply_rotary_pos_emb(k, cos.unsqueeze(1), sin.unsqueeze(1))

        kv_cache.write(self.layer_idx, slot_mapping, k, v)
        k_all, v_all = kv_cache.gather(self.layer_idx, block_table.physical_slots())
        k_all = self._repeat_kv(k_all)  # (L, H, D)
        v_all = self._repeat_kv(v_all)

        q4 = q.permute(1, 0, 2).unsqueeze(0)
        k4 = k_all.permute(1, 0, 2).unsqueeze(0)
        v4 = v_all.permute(1, 0, 2).unsqueeze(0)
        mask = make_causal_mask(L, L).to(x.device)
        out, _ = scaled_dot_product_attention(q4, k4, v4, mask)
        out = out.squeeze(0).permute(1, 0, 2).reshape(L, -1)
        return self.o_proj(out)

    def decode_batch(self, x, cos, sin, kv_cache, slot_mapping, block_tables):
        """一批请求各走一步。x: (B, E)。返回 (B, E)。"""
        B = x.shape[0]
        q, k_new, v_new = self._project_qkv(x)  # (B, H, D) / (B, Hkv, D)
        q = apply_rotary_pos_emb(q, cos.unsqueeze(1), sin.unsqueeze(1))
        k_new = apply_rotary_pos_emb(k_new, cos.unsqueeze(1), sin.unsqueeze(1))

        kv_cache.write(self.layer_idx, slot_mapping, k_new, v_new)
        slot_lists = [bt.physical_slots() for bt in block_tables]
        k_all, v_all, pad_mask = kv_cache.gather_padded(self.layer_idx, slot_lists)
        # (B, Lmax, Hkv, D) -> (B, Hkv, Lmax, D) -> GQA 复制 -> (B, H, Lmax, D)
        k_all = self._repeat_kv(k_all.transpose(1, 2))
        v_all = self._repeat_kv(v_all.transpose(1, 2))
        q4 = q.unsqueeze(2)  # (B, H, 1, D)
        mask = pad_mask.view(B, 1, 1, -1)
        out, _ = scaled_dot_product_attention(q4, k_all, v_all, mask)
        out = out.squeeze(2).reshape(B, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU：down_proj( silu(gate_proj(x)) * up_proj(x) )。"""

    def __init__(self, config):
        super().__init__()
        E = config.hidden_size
        I = config.intermediate_size
        self.gate_proj = nn.Linear(E, I, bias=False)
        self.up_proj = nn.Linear(E, I, bias=False)
        self.down_proj = nn.Linear(I, E, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Qwen3DecoderLayer(nn.Module):
    """一层 Qwen3：RMSNorm → GQA 注意力 → 残差 → RMSNorm → SwiGLU → 残差。"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(config)

    def naive_forward(self, x, cos, sin):
        x = x + self.self_attn.naive_forward(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def prefill(self, x, cos, sin, kv_cache, slot_mapping, block_table):
        x = x + self.self_attn.prefill(
            self.input_layernorm(x), cos, sin, kv_cache, slot_mapping, block_table
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def decode_batch(self, x, cos, sin, kv_cache, slot_mapping, block_tables):
        x = x + self.self_attn.decode_batch(
            self.input_layernorm(x), cos, sin, kv_cache, slot_mapping, block_tables
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3ForCausalLM(nn.Module):
    """完整的 Qwen3：嵌入 → N 层 → RMSNorm → lm_head（与嵌入共享权重）。

    与 MiniTransformer 一样实现了引擎接口：prefill / decode_batch。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    # ---- 朴素整段前向（Week 6 与 HF 对拍用）----

    def forward(self, input_ids):
        """朴素整段前向（对拍/教学用）。input_ids: (1, L)。返回 logits (1, L, vocab)。"""
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "朴素前向一次只看一条序列"
        input_ids = input_ids[0]
        L = input_ids.shape[0]
        positions = torch.arange(L, device=input_ids.device)
        cos, sin = self.rotary_emb(positions)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer.naive_forward(x, cos, sin)
        return self.lm_head(self.norm(x)).unsqueeze(0)

    # ---- 分页引擎接口 ----

    @torch.no_grad()
    def prefill(self, input_ids, kv_cache, slot_mapping, block_table):
        """一个请求的 prefill。input_ids: (L,)。返回最后位置的 logits (vocab,)。"""
        L = input_ids.shape[0]
        positions = torch.arange(L, device=input_ids.device)
        cos, sin = self.rotary_emb(positions)
        cos = cos.to(self.embed_tokens.weight.dtype)
        sin = sin.to(self.embed_tokens.weight.dtype)
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer.prefill(x, cos, sin, kv_cache, slot_mapping, block_table)
        return self.lm_head(self.norm(x[-1]))

    @torch.no_grad()
    def decode_batch(self, token_ids, positions, kv_cache, slot_mapping, block_tables):
        """一批请求各走一步。token_ids/positions: (B,)。返回 (B, vocab)。"""
        cos, sin = self.rotary_emb(positions)
        cos = cos.to(self.embed_tokens.weight.dtype)
        sin = sin.to(self.embed_tokens.weight.dtype)
        x = self.embed_tokens(token_ids)
        for layer in self.layers:
            x = layer.decode_batch(x, cos, sin, kv_cache, slot_mapping, block_tables)
        return self.lm_head(self.norm(x))

    # ---- 权重加载 ----

    @classmethod
    def from_pretrained(cls, model_path, device="cpu", dtype=torch.float32):
        """从本地 HF checkpoint 目录加载（config.json + model.safetensors）。"""
        from safetensors.torch import load_file

        with open(os.path.join(model_path, "config.json")) as f:
            config = Qwen3Config(json.load(f))
        model = cls(config)
        state = {}
        for fname in sorted(os.listdir(model_path)):
            if fname.endswith(".safetensors"):
                state.update(load_file(os.path.join(model_path, fname)))
        # HF 的键以 "model." 开头，我们的模块就是按这个结构命名的，
        # 剥掉前缀即可对上；lm_head 与嵌入共享权重——有的 checkpoint 存了它
        # （Qwen3-0.6B 就存了），有的没存，两种都能兼容。
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if k != "lm_head.weight"]
        missing = [k for k in missing if k != "lm_head.weight"]
        assert not unexpected, f"checkpoint 里多了不认识的权重: {unexpected}"
        assert not missing, f"checkpoint 缺了权重: {missing}"
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        return model.to(device=device, dtype=dtype)
