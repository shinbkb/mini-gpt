import pytest
import torch
import torch.nn.functional as F

from src.models.gpt import GPT, GPTConfig
from src.models.moe import Top2MoE


def tiny_moe_config(**overrides):
    values = dict(vocab_size=32, block_size=8, n_layers=2, d_model=16,
                  n_heads=2, d_ff=32, dropout=0.0, model_type="moe",
                  n_experts=4, top_k=2, auxiliary_loss_coefficient=0.01)
    values.update(overrides)
    return GPTConfig(**values)


def test_top2_indices_and_weights_normalize():
    moe = Top2MoE(8, 16, n_experts=4, top_k=2)
    x = torch.randn(2, 5, 8)
    _, aux = moe(x)
    probs = moe.router(x.reshape(-1, 8)).softmax(-1)
    top, indices = probs.topk(2, dim=-1)
    assert indices.shape == (10, 2)
    torch.testing.assert_close((top / top.sum(-1, keepdim=True)).sum(-1), torch.ones(10))
    assert aux.ndim == 0 and torch.isfinite(aux)
    assert moe.last_routing_stats["token_counts"].sum().item() == 20


def test_moe_shapes_and_gradients_reach_router_and_selected_experts():
    torch.manual_seed(1)
    moe = Top2MoE(8, 16, n_experts=4, top_k=2)
    x = torch.randn(3, 6, 8, requires_grad=True)
    output, aux = moe(x)
    (output.square().mean() + aux).backward()
    assert output.shape == x.shape
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert any(p.grad is not None for expert in moe.experts for p in expert.parameters())
    assert torch.isfinite(aux)


def test_moe_gpt_forward_loss_and_counts():
    model = GPT(tiny_moe_config())
    ids = torch.randint(0, 32, (2, 8))
    logits, aux = model(ids, return_aux_loss=True)
    loss = F.cross_entropy(logits.reshape(-1, 32), ids.reshape(-1)) + aux
    loss.backward()
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(loss)
    assert model.parameter_count() > model.activated_parameter_count() > 0


def test_moe_checkpoint_save_load(tmp_path):
    torch.manual_seed(7)
    model = GPT(tiny_moe_config()).eval()
    ids = torch.randint(0, 32, (1, 5))
    expected = model(ids)
    path = tmp_path / "moe.pt"
    torch.save({"model_config": model.config.__dict__, "model_state": model.state_dict()}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = GPT(GPTConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model_state"])
    torch.testing.assert_close(restored.eval()(ids), expected)


def test_invalid_moe_config():
    with pytest.raises(ValueError):
        tiny_moe_config(top_k=5)


def test_legacy_dense_config_defaults_to_dense():
    config = GPTConfig.from_mapping({"vocab_size": 32, "block_size": 8,
                                    "n_layers": 1, "d_model": 16,
                                    "n_heads": 2, "d_ff": 32,
                                    "dropout": 0.0})
    assert config.model_type == "dense"
    assert GPT(config).parameter_count() > 0
