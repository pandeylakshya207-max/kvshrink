"""Qwen2 forward pass and greedy decoding over a custom KV cache.

Only the Hugging Face weights are used; RoPE, GQA attention, the causal mask
and the cache are implemented here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from kvshrink.cache import KVCache, StaticKVCache


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm computed in float32, cast back to the input dtype (matches HF Qwen2)."""
    dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class QwenRunner:
    """Runs a Hugging Face Qwen2ForCausalLM with a kvshrink cache."""

    def __init__(self, hf_model) -> None:
        cfg = hf_model.config
        rope = getattr(cfg, "rope_parameters", None)
        if rope and rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope['rope_type']} not supported")
        if getattr(cfg, "use_sliding_window", False):
            raise NotImplementedError("sliding-window attention not supported")
        self.cfg = cfg
        self.body = hf_model.model
        self.lm_head = hf_model.lm_head
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        theta = rope["rope_theta"] if rope else cfg.rope_theta
        device = self.lm_head.weight.device
        exps = torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim
        self.inv_freq = (1.0 / (theta ** exps)).to(device)

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    def new_cache(self, batch_size: int, max_len: int) -> StaticKVCache:
        """Full-precision cache sized for this model."""
        return StaticKVCache.from_config(self.cfg, batch_size, max_len, self.dtype, self.device)

    def _rope(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _attention_with_probs(
        self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit GQA attention that also returns attention mass per KV head.

        The last T keys are the new tokens (causal among themselves); all earlier
        keys are stored past and fully visible. Returns (out [B, H_q, T, D],
        attn [B, H_kv, S]) where attn sums the probabilities over the query heads
        sharing each KV head and over the T queries.
        """
        B, S = q.shape[0], K.shape[2]
        rep = self.n_heads // self.n_kv
        qg = q.reshape(B, self.n_kv, rep * T, self.head_dim)
        scores = (qg.float() @ K.float().transpose(-1, -2)) * self.head_dim ** -0.5
        scores = scores.view(B, self.n_kv, rep, T, S)
        q_idx = torch.arange(S - T, S, device=q.device)
        k_idx = torch.arange(S, device=q.device)
        scores = scores.masked_fill(k_idx[None, :] > q_idx[:, None], float("-inf"))
        probs = scores.softmax(dim=-1)
        out = probs.view(B, self.n_kv, rep * T, S).to(V.dtype) @ V
        out = out.view(B, self.n_kv, rep, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
        return out, probs.sum(dim=(2, 3))

    def _attention(self, attn, h, cos, sin, cache: KVCache, layer: int) -> torch.Tensor:
        B, T, _ = h.shape
        q = attn.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        K, V = cache.update(layer, k, v)
        rep = self.n_heads // self.n_kv
        if getattr(cache, "needs_attention", False):
            out, mass = self._attention_with_probs(q, K, V, T)
            cache.observe(layer, mass)
        elif T == 1:
            # Decode: fold the `rep` query heads sharing a KV head into the query
            # axis, so K/V are never copied (memory stays proportional to the cache).
            qg = q.reshape(B, self.n_kv, rep, self.head_dim)
            out = F.scaled_dot_product_attention(qg, K, V)
            out = out.reshape(B, self.n_heads, 1, self.head_dim)
        else:
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
            if K.shape[2] == T:
                out = F.scaled_dot_product_attention(q, K, V, is_causal=True)
            else:
                # Stored past (possibly evicted, so fewer than the positions seen) is
                # fully visible; the T new tokens are causal among themselves.
                q_pos = torch.arange(K.shape[2] - T, K.shape[2], device=h.device)
                k_pos = torch.arange(K.shape[2], device=h.device)
                mask = k_pos[None, :] <= q_pos[:, None]
                out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return attn.o_proj(out)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, cache: KVCache, last_only: bool = False) -> torch.Tensor:
        """Run new tokens [B, T] through the model, appending to `cache`.

        RoPE positions continue from `cache.tokens_seen()`, which can exceed the
        stored length when the cache evicts. Returns logits [B, T, vocab], or
        [B, 1, vocab] if `last_only`.
        """
        past = cache.tokens_seen(0)
        T = input_ids.shape[1]
        positions = torch.arange(past, past + T, device=input_ids.device)
        x = self.body.embed_tokens(input_ids)
        cos, sin = self._rope(positions, x.dtype)
        for i, layer in enumerate(self.body.layers):
            h = rms_norm(x, layer.input_layernorm.weight, self.eps)
            x = x + self._attention(layer.self_attn, h, cos, sin, cache, i)
            h = rms_norm(x, layer.post_attention_layernorm.weight, self.eps)
            mlp = layer.mlp
            x = x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))
        if last_only:
            x = x[:, -1:]
        x = rms_norm(x, self.body.norm.weight, self.eps)
        return self.lm_head(x)

    @torch.no_grad()
    def greedy_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, cache: KVCache | None = None
    ) -> torch.Tensor:
        """Greedy decoding without EOS stopping. Returns [B, T_prompt + max_new_tokens]."""
        if cache is None:
            cache = self.new_cache(input_ids.shape[0], input_ids.shape[1] + max_new_tokens)
        logits = self.forward(input_ids, cache, last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = [input_ids, nxt]
        for _ in range(max_new_tokens - 1):
            logits = self.forward(nxt, cache, last_only=True)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
        return torch.cat(out, dim=1)
