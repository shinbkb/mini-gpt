import torch
import pytest

from src.models.gpt import GPT, GPTConfig
from src.models.rope import apply_rope


def tiny_config(**overrides):
    values = dict(vocab_size=32, block_size=16, n_layers=1,
                  d_model=16, n_heads=2, d_ff=32, dropout=0.0)
    values.update(overrides)
    return GPTConfig(**values)


def test_model_shape_and_gradients():
    model = GPT(tiny_config())
    tokens = torch.randint(0, 32, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 32)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_future_token_does_not_change_prior_logits():
    torch.manual_seed(7)
    model = GPT(tiny_config()).eval()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed = original.clone()
    changed[0, 4] = 8
    with torch.no_grad():
        logits_a, logits_b = model(original), model(changed)
    torch.testing.assert_close(logits_a[:, :4], logits_b[:, :4], rtol=1e-5, atol=1e-6)


def test_rope_shapes_and_invalid_inputs():
    x = torch.randn(2, 4, 7, 8)
    assert apply_rope(x).shape == x.shape
    assert not torch.equal(apply_rope(x), x)
    with pytest.raises(ValueError):
        apply_rope(torch.randn(2, 7, 8))
    with pytest.raises(ValueError):
        apply_rope(torch.randn(1, 1, 2, 7))


def test_invalid_model_inputs_and_generation():
    model = GPT(tiny_config()).eval()
    with pytest.raises(ValueError):
        model(torch.ones(1, 2, dtype=torch.int32))
    with pytest.raises(ValueError):
        model(torch.ones(1, 17, dtype=torch.long))
    generated = model.generate(torch.tensor([[1, 2]]), 4, top_k=3,
                               generator=torch.Generator().manual_seed(1))
    assert generated.shape == (1, 6)


def test_tiny_batch_overfit():
    torch.manual_seed(1)
    model = GPT(tiny_config(vocab_size=8, block_size=4, d_model=16, n_heads=2, d_ff=32)).train()
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, 5]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    initial = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert final < initial * 0.2


def test_checkpoint_reload_and_evaluate(tmp_path):
    from src.training.trainer import train
    config = tmp_path / "tiny.yaml"
    config.write_text('''seed: 2\nmodel:\n  vocab_size: 16\n  block_size: 4\n  n_layers: 1\n  d_model: 16\n  n_heads: 2\n  d_ff: 32\n  dropout: 0.0\ntraining:\n  batch_size: 2\n  max_steps: 2\n  eval_interval: 1\n  eval_batches: 1\n  learning_rate: 0.001\n  precision: fp32\n''')
    torch.save(torch.randint(0, 16, (80,), dtype=torch.long), tmp_path / "train.pt")
    torch.save(torch.randint(0, 16, (40,), dtype=torch.long), tmp_path / "val.pt")
    output = tmp_path / "out"
    summary = train(config, train_tokens_path=tmp_path / "train.pt",
                    validation_tokens_path=tmp_path / "val.pt", output_dir=output,
                    max_steps=2, device_name="cpu")
    assert summary["step"] == 2 and summary["val_loss"] is not None
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    assert model(torch.ones((1, 4), dtype=torch.long)).shape == (1, 4, 16)
    assert (output / "best.pt").exists() and (output / "metrics.jsonl").exists()
    resumed = train(config, train_tokens_path=tmp_path / "train.pt",
                    validation_tokens_path=tmp_path / "val.pt", output_dir=output,
                    max_steps=3, device_name="cpu", resume=output / "last.pt")
    assert resumed["step"] == 3
