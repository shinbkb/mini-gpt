"""M0 CPU checks; no dataset downloads or model training."""
import importlib
import random
from pathlib import Path

import pytest
import torch
import yaml

from src.environment import environment_info, seed_everything


def test_package_imports():
    for name in ("src", "src.tokenizer", "src.data", "src.models", "src.inference", "src.training"):
        assert importlib.import_module(name) is not None


def test_cpu_autograd():
    x = torch.tensor([1., 2., 3.], device="cpu", requires_grad=True)
    loss = x.square().sum()
    loss.backward()
    assert loss.item() == 14.
    torch.testing.assert_close(x.grad, torch.tensor([2., 4., 6.]))


def test_seed_reproducibility():
    seed_everything(42)
    first = (random.random(), torch.rand(4))
    seed_everything(42)
    assert random.random() == first[0]
    assert torch.equal(torch.rand(4), first[1])
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_cpu_environment_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = environment_info(123)
    assert info["device"] == "cpu"
    assert info["gpu_devices"] == []
    assert info["seed"] == 123
    assert info["libraries"]["torch"]


@pytest.mark.parametrize("name", ["dense", "moe"])
def test_target_configs(name):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())
    model = config["model"]
    assert model["d_model"] == model["n_heads"] * model["head_dim"]
    assert model["vocab_size"] == 8000 and model["block_size"] == 512
    assert config["seed"] == 42
    if name == "moe":
        assert model["top_k"] == 2 <= model["n_experts"] == 4
