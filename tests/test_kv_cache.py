import pytest
import torch

from src.inference.kv_cache import KVCache
from src.models.gpt import GPT, GPTConfig


def tiny_model():
    torch.manual_seed(19)
    return GPT(GPTConfig(vocab_size=41, block_size=16, n_layers=3,
                         d_model=24, n_heads=3, d_ff=48, dropout=0.0)).eval()


def test_kv_cache_update_shapes_dtype_device_and_clear():
    cache = KVCache(2)
    key, value = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
    entry = cache.update(0, key, value)
    assert len(cache) == 4
    assert entry.key.shape == (2, 3, 4, 8)
    next_entry = cache.update(0, key[:, :, :1], value[:, :, :1])
    assert len(cache) == 5 and next_entry.key.shape[-2] == 5
    assert next_entry.key.dtype == key.dtype and next_entry.key.device == key.device
    cache.clear()
    assert len(cache) == 0
    cache.update(0, key, value)
    with pytest.raises(ValueError):
        cache.update(0, key.to(torch.float64), value.to(torch.float64))


def test_cached_incremental_logits_match_full_forward_with_rope_offsets():
    model = tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (2, 11))
    cache = KVCache(model.config.n_layers)
    with torch.inference_mode():
        full_logits = model(tokens)
        cached_prefill = model(tokens[:, :7], cache=cache)
        torch.testing.assert_close(cached_prefill, full_logits[:, :7], rtol=1e-5, atol=1e-6)
        for position in range(7, tokens.shape[1]):
            step_logits = model(tokens[:, position:position + 1], cache=cache)
            torch.testing.assert_close(step_logits[:, -1], full_logits[:, position],
                                       rtol=2e-5, atol=2e-6)
    assert len(cache) == tokens.shape[1]
    assert all(layer.key.shape == (2, 3, 11, 8) for layer in cache.layers)


def test_forward_use_cache_returns_reusable_cache():
    model = tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 4))
    with torch.inference_mode():
        logits, cache = model(tokens, use_cache=True)
        next_logits, same_cache = model(tokens[:, -1:], use_cache=True, cache=cache)
    assert logits.shape == (1, 4, model.config.vocab_size)
    assert next_logits.shape == (1, 1, model.config.vocab_size)
    assert same_cache is cache and len(cache) == 5


def test_generate_cache_and_no_cache_greedy_match():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    cached = model.generate(prompt, 8, temperature=1.0, top_k=1, use_cache=True)
    uncached = model.generate(prompt, 8, temperature=1.0, top_k=1, use_cache=False)
    assert torch.equal(cached, uncached)


def test_generate_cache_and_no_cache_match_across_sliding_context_boundary():
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    cached = model.generate(prompt, 10, temperature=1.0, top_k=1, use_cache=True)
    uncached = model.generate(prompt, 10, temperature=1.0, top_k=1, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cache_context_limit_and_decode_chunk_guard():
    model = tiny_model()
    cache = KVCache(model.config.n_layers)
    with torch.inference_mode():
        model(torch.ones((1, 15), dtype=torch.long), cache=cache)
        with pytest.raises(ValueError, match="block_size"):
            model(torch.ones((1, 2), dtype=torch.long), cache=cache)
        cache.clear()
        model(torch.ones((1, 2), dtype=torch.long), cache=cache)
        with pytest.raises(ValueError, match="one new token"):
            model(torch.ones((1, 2), dtype=torch.long), cache=cache)


def test_kv_cache_invalid_dimensions():
    with pytest.raises(ValueError):
        KVCache(0)
    with pytest.raises(IndexError):
        KVCache(1).update(1, torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))


def test_kv_benchmark_skips_unsupported_contexts():
    from experiments.benchmark_kv import benchmark
    model = GPT(GPTConfig(vocab_size=17, block_size=8, n_layers=1,
                          d_model=8, n_heads=2, d_ff=16, dropout=0.0)).eval()
    rows = benchmark(model, contexts=[4, 16], new_tokens=3, warmup=0, repeats=1,
                     precision="fp32", device=torch.device("cpu"))
    assert len(rows) == 1 and rows[0]["context_length"] == 4
    assert rows[0]["skipped_contexts"] == [16]
    assert rows[0]["cache"]["generated_tokens"] == 3
    assert rows[0]["cache"]["tokens_per_second"] > 0
