"""Dense/MoE GPT trainer with validation and resumable checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F
import yaml

from src.environment import environment_info, seed_everything
from src.models.gpt import GPT, GPTConfig


def load_tokens(path: str | Path, vocab_size: int) -> torch.Tensor:
    """Load a one-dimensional CPU token tensor saved by M1 and validate IDs."""
    tokens = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError(f"{path} must contain a rank-1 torch.long tensor")
    if tokens.numel() < 2 or int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
        raise ValueError(f"{path} contains too few tokens or IDs outside vocabulary")
    return tokens.contiguous()


def _batch(tokens: torch.Tensor, batch_size: int, block_size: int,
           device: torch.device, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - block_size
    if max_start <= 0:
        raise ValueError(f"need at least block_size + 1 ({block_size + 1}) tokens")
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    indices = starts[:, None] + offsets[None, :]
    x = tokens[indices].to(device, non_blocking=device.type == "cuda")
    y = tokens[indices + 1].to(device, non_blocking=device.type == "cuda")
    return x, y


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _ppl(loss: float) -> float | None:
    return math.exp(loss) if math.isfinite(loss) and loss < 80 else None


@torch.no_grad()
def evaluate_loss(model: GPT, tokens: torch.Tensor, *, batch_size: int,
                  eval_batches: int, device: torch.device, seed: int) -> float:
    """Estimate mean next-token cross-entropy on random validation windows."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    for _ in range(eval_batches):
        x, y = _batch(tokens, batch_size, model.config.block_size, device, generator)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _save_checkpoint(path: Path, *, model: GPT, optimizer, scheduler, scaler,
                     step: int, best_val: float, config: dict, device: torch.device) -> None:
    checkpoint = {
        "format_version": 1,
        "model_config": model.config.__dict__,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "best_val_loss": best_val,
        "run_config": config,
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def train(config_path: str | Path, *, train_tokens_path: str | Path | None = None,
          validation_tokens_path: str | Path | None = None, output_dir: str | Path | None = None,
          max_steps: int | None = None, device_name: str = "auto",
          resume: str | Path | None = None, overfit_batch: bool = False) -> dict:
    """Train Dense or MoE GPT. ``overfit_batch`` repeats one sampled batch."""
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError("config must contain a model mapping")
    model_config = GPTConfig.from_mapping(raw["model"])
    training = raw.get("training", {})
    seed = int(raw.get("seed", 42))
    seed_everything(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    requested_precision = str(training.get("precision", "fp32")).lower()
    if requested_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    precision = requested_precision if device.type == "cuda" else "fp32"

    train_path = Path(train_tokens_path or raw.get("data", {}).get("train_tokens", "artifacts/train_tokens.pt"))
    val_path = Path(validation_tokens_path or raw.get("data", {}).get("validation_tokens", "artifacts/validation_tokens.pt"))
    train_tokens = load_tokens(train_path, model_config.vocab_size)
    val_tokens = load_tokens(val_path, model_config.vocab_size)
    batch_size = int(training.get("batch_size", 8))
    steps = int(max_steps if max_steps is not None else training.get("max_steps", 1000))
    eval_interval = int(training.get("eval_interval", 100))
    eval_batches = int(training.get("eval_batches", 20))
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    if min(batch_size, steps, eval_interval, eval_batches, grad_accum) <= 0:
        raise ValueError("batch size, steps, eval settings and accumulation must be positive")
    learning_rate = float(training.get("learning_rate", 3e-4))
    min_lr_ratio = float(training.get("min_lr_ratio", 0.1))
    warmup_steps = int(training.get("warmup_steps", min(50, steps // 10)))
    weight_decay = float(training.get("weight_decay", 0.1))
    grad_clip = float(training.get("gradient_clip", 1.0))
    destination = Path(output_dir or training.get("output_dir", "runs/dense"))
    destination.mkdir(parents=True, exist_ok=True)

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    def lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(steps - warmup_steps, 1)))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    start_step, best_val = 0, math.inf
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 1 or checkpoint.get("model_config") != model_config.__dict__:
            raise ValueError("checkpoint format or model config does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint["best_val_loss"])
        random.setstate(checkpoint["rng_python"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if device.type == "cuda" and checkpoint.get("rng_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        if start_step >= steps:
            raise ValueError("checkpoint step is already >= requested max_steps")

    config_record = {"model": model_config.__dict__, "training": training,
                     "seed": seed, "device": str(device), "precision": precision,
                     "train_tokens": str(train_path), "validation_tokens": str(val_path),
                     "train_tokens_sha256": __import__('hashlib').sha256(train_path.read_bytes()).hexdigest(),
                     "validation_tokens_sha256": __import__('hashlib').sha256(val_path.read_bytes()).hexdigest(),
                     "environment": environment_info(seed)}
    metrics_path = destination / "metrics.jsonl"
    fixed_batch = _batch(train_tokens, batch_size, model_config.block_size, device) if overfit_batch else None
    if not resume:
        metrics_path.write_text("", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    training_seconds = 0.0
    last_eval = None
    for step in range(start_step, steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_loss_value = 0.0
        for _ in range(grad_accum):
            x, y = fixed_batch if fixed_batch is not None else _batch(train_tokens, batch_size, model_config.block_size, device)
            with _autocast(device, precision):
                logits, auxiliary_loss = model(x, return_aux_loss=True)
                language_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = (language_loss + auxiliary_loss) / grad_accum
            scaler.scale(loss).backward()
            train_loss_value += loss.detach().item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()
        # GradScaler lowers the scale when it skipped optimizer.step() due to
        # non-finite fp16 gradients; do not advance LR schedule in that case.
        if new_scale >= old_scale:
            scheduler.step()
        training_seconds += time.perf_counter() - step_started

        completed = step + 1
        if completed % eval_interval == 0 or completed == steps or completed == start_step + 1:
            val_loss = evaluate_loss(model, val_tokens, batch_size=batch_size,
                                     eval_batches=eval_batches, device=device, seed=seed + completed)
            last_eval = val_loss
            elapsed = max(training_seconds, 1e-9)
            routing = [block.feed_forward.last_routing_stats
                       for block in model.blocks
                       if hasattr(block.feed_forward, "last_routing_stats") and block.feed_forward.last_routing_stats]
            routing_summary = {}
            if routing:
                routing_summary = {
                    "expert_token_counts": torch.stack([item["token_counts"].float() for item in routing]).sum(0).cpu().tolist(),
                    "mean_expert_utilization": torch.stack([item["expert_utilization"] for item in routing]).mean().item(),
                    "mean_router_entropy": torch.stack([item["router_entropy"] for item in routing]).mean().item(),
                    "auxiliary_loss": float(auxiliary_loss.detach().item()),
                }
            record = {"step": completed, "train_loss": train_loss_value,
                      "val_loss": val_loss, "val_perplexity": _ppl(val_loss),
                      "tokens_per_second": completed * batch_size * model_config.block_size * grad_accum / elapsed,
                      "training_seconds": training_seconds,
                      "learning_rate": scheduler.get_last_lr()[0],
                      "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                      "parameters_total": model.parameter_count(),
                      "parameters_activated": model.activated_parameter_count(),
                      **routing_summary}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            best_val = min(best_val, val_loss)
            _save_checkpoint(destination / "last.pt", model=model, optimizer=optimizer,
                             scheduler=scheduler, scaler=scaler, step=completed,
                             best_val=best_val, config=config_record, device=device)
            if val_loss <= best_val:
                shutil.copy2(destination / "last.pt", destination / "best.pt")
    return {"output_dir": str(destination), "step": steps, "val_loss": last_eval,
            "val_perplexity": _ppl(last_eval) if last_eval is not None else None,
            "parameters": model.parameter_count(),
            "activated_parameters": model.activated_parameter_count(),
            "device": str(device), "precision": precision}
