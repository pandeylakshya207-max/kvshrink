import os

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvshrink.model import QwenRunner


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return Qwen2ForCausalLM(cfg).eval()


@torch.no_grad()
def hf_greedy(model, ids: torch.Tensor, n: int) -> torch.Tensor:
    """Reference: HF forward with its own default cache, no EOS stopping."""
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [ids, nxt]
    for _ in range(n - 1):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(nxt)
    return torch.cat(toks, dim=1)


def test_tiny_prefill_logits_match():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))
    ours = runner.forward(ids, runner.new_cache(2, 12))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


def test_tiny_greedy_tokens_match():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(2))
    assert torch.equal(runner.greedy_generate(ids, 30), hf_greedy(model, ids, 30))


def test_chunked_prefill_matches_single_prefill():
    model = tiny_model()
    runner = QwenRunner(model)
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(3))
    full = runner.forward(ids, runner.new_cache(1, 16))
    cache = runner.new_cache(1, 16)
    a = runner.forward(ids[:, :9], cache)
    b = runner.forward(ids[:, 9:], cache)
    assert torch.allclose(torch.cat([a, b], dim=1), full, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("KVSHRINK_SLOW") != "1", reason="set KVSHRINK_SLOW=1")
def test_qwen05b_greedy_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    msgs = [{"role": "user", "content": "Explain what a KV cache is in two sentences."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    ours = QwenRunner(model).greedy_generate(ids, 64)
    ref = hf_greedy(model, ids, 64)
    assert torch.equal(ours, ref), tok.decode(ours[0, ids.shape[1]:])
