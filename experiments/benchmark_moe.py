"""Compare Dense and MoE checkpoints under the same evaluation settings."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.environment import environment_info
from src.models.gpt import GPT, GPTConfig
from src.training.trainer import load_tokens, _ppl


def _load(path: str, device: torch.device) -> GPT:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    return model


@torch.inference_mode()
def _measure(model: GPT, tokens: torch.Tensor, *, batch_size: int, batches: int,
             repeats: int, device: torch.device, precision: str,
             generation_tokens: int = 32) -> dict:
    block = model.config.block_size
    if tokens.numel() < block + 1:
        raise ValueError("validation token stream is shorter than block_size + 1")
    generator = torch.Generator().manual_seed(314159)
    starts = torch.randint(tokens.numel() - block, (batches * batch_size,), generator=generator)
    offsets = torch.arange(block)
    xs = torch.stack([tokens[i:i + block] for i in starts]).reshape(batches, batch_size, block)
    ys = torch.stack([tokens[i + 1:i + block + 1] for i in starts]).reshape(batches, batch_size, block)
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    losses, times = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        run_losses = []
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for idx in range(batches):
            x, y = xs[idx].to(device), ys[idx].to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x)
            run_losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        if device.type == "cuda": torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        losses.append(sum(run_losses) / len(run_losses))
    elapsed = statistics.median(times)
    val_loss = sum(losses) / len(losses)
    prompt_len = min(64, block)
    prompt = xs[0, :1, :prompt_len].to(device)
    generation_times = []
    for _ in range(repeats):
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            generated = model.generate(prompt, generation_tokens, temperature=1.0, use_cache=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        generation_times.append(time.perf_counter() - start)
    generation_elapsed = statistics.median(generation_times)
    routing = [block.feed_forward.last_routing_stats for block in model.blocks
               if hasattr(block.feed_forward, "last_routing_stats") and block.feed_forward.last_routing_stats]
    stats = {}
    if routing:
        counts = torch.stack([item["token_counts"].float() for item in routing]).sum(0)
        stats = {"expert_token_counts": counts.cpu().tolist(),
                 "routing_distribution": (counts / counts.sum()).cpu().tolist(),
                 "mean_expert_utilization": torch.stack([item["expert_utilization"] for item in routing]).mean().item(),
                 "mean_router_entropy": torch.stack([item["router_entropy"] for item in routing]).mean().item()}
    return {"validation_loss": val_loss, "validation_perplexity": _ppl(val_loss),
            "validation_tokens_per_second": batches * batch_size * block / elapsed,
            "generation_prompt_tokens": prompt_len,
            "generation_new_tokens": generation_tokens,
            "generation_tokens_per_second": generation_tokens / max(generation_elapsed, 1e-12),
            "generation_seconds": generation_elapsed,
            "sample_generation": generated[0, prompt_len:].cpu().tolist(),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "parameters_total": model.parameter_count(),
            "parameters_activated": model.activated_parameter_count(), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--moe-checkpoint", required=True)
    parser.add_argument("--validation-tokens", default="artifacts/validation_tokens.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--train-tokens-per-model", type=int)
    parser.add_argument("--dense-training-metrics")
    parser.add_argument("--moe-training-metrics")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="reports/benchmark_moe.json")
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.repeats) < 1:
        parser.error("batch-size, batches, and repeats must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    dense, moe = _load(args.dense_checkpoint, device), _load(args.moe_checkpoint, device)
    if dense.config.vocab_size != moe.config.vocab_size or dense.config.block_size != moe.config.block_size:
        parser.error("Dense and MoE checkpoints must use the same vocabulary and block_size")
    tokens = load_tokens(args.validation_tokens, dense.config.vocab_size)
    results = {}
    for name, model, path in (("dense", dense, args.dense_checkpoint), ("moe", moe, args.moe_checkpoint)):
        results[name] = {"checkpoint": str(path), **_measure(model, tokens, batch_size=args.batch_size,
                         batches=args.batches, repeats=args.repeats, device=device,
                         precision=args.precision, generation_tokens=args.generation_tokens)}
    for name, metrics_path in (("dense", args.dense_training_metrics),
                               ("moe", args.moe_training_metrics)):
        if metrics_path:
            metrics = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
            results[name]["training_metrics"] = metrics[-1] if metrics else None
    if args.train_tokens_per_model is not None:
        results["training_tokens_per_model"] = args.train_tokens_per_model
    report = {"device": str(device), "precision": args.precision, "batch_size": args.batch_size,
              "context_length": dense.config.block_size, "validation_batches": args.batches,
              "repeats": args.repeats, "environment": environment_info(314159), "results": results}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".csv").write_text(
        "model,validation_loss,validation_perplexity,validation_tokens_per_second,generation_tokens_per_second,peak_vram_bytes,parameters_total,parameters_activated\n" +
        "".join(f"{name},{row['validation_loss']},{row['validation_perplexity']},{row['validation_tokens_per_second']},{row['generation_tokens_per_second']},{row['peak_vram_bytes']},{row['parameters_total']},{row['parameters_activated']}\n" for name, row in results.items() if isinstance(row, dict) and "validation_loss" in row),
        encoding="utf-8")
    moe_result = results.get("moe", {})
    distribution = moe_result.get("routing_distribution", [])
    if distribution:
        width, height, left, top, bottom = 640, 360, 60, 50, 60
        plot_h = height - top - bottom
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 '<text x="320" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">MoE routing distribution</text>']
        for tick in range(5):
            fraction = tick / 4
            y = top + plot_h * (1 - fraction)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="610" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{fraction:.2f}</text>')
        bar_w = min(72, 480 / max(1, len(distribution)))
        for index, fraction in enumerate(distribution):
            x = left + (index + 0.5) * 480 / len(distribution) - bar_w / 2
            bar_h = plot_h * fraction
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#2474b5"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">Expert {index}</text>')
        parts.append('</svg>')
        output.with_suffix(".svg").write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
