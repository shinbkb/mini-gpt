"""Compare Dense GPT decode latency with and without per-layer KV caching."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from src.inference.kv_cache import KVCache
from src.environment import environment_info
from src.models.gpt import GPT, GPTConfig


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _run(model: GPT, prompt: torch.Tensor, new_tokens: int, use_cache: bool,
         device: torch.device) -> dict:
    cache = KVCache(model.config.n_layers) if use_cache else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    logits = model(prompt, cache=cache) if use_cache else model(prompt)
    _sync(device)
    first_token_seconds = time.perf_counter() - started
    generated = torch.cat((prompt, logits[:, -1].argmax(-1, keepdim=True)), dim=1)

    _sync(device)
    started = time.perf_counter()
    for _ in range(max(0, new_tokens - 1)):
        if use_cache:
            if len(cache) >= model.config.block_size:
                # Match model.generate's sliding-window behavior once context is full.
                cache.clear()
                logits = model(generated[:, -model.config.block_size:], cache=cache)
            else:
                logits = model(generated[:, -1:], cache=cache)
        else:
            logits = model(generated[:, -model.config.block_size:])
        generated = torch.cat((generated, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    _sync(device)
    decode_seconds = time.perf_counter() - started
    return {
        "first_token_latency_ms": first_token_seconds * 1000,
        "next_token_latency_ms": (decode_seconds / max(1, new_tokens - 1)) * 1000,
        "tokens_per_second": max(0, new_tokens - 1) / max(decode_seconds, 1e-12),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "generated_tokens": new_tokens,
    }


def benchmark(model: GPT, *, contexts: list[int], new_tokens: int = 64,
              warmup: int = 2, repeats: int = 5, seed: int = 42,
              precision: str = "fp32", device: torch.device | None = None) -> list[dict]:
    """Measure only context lengths supported by the configured model."""
    device = device or next(model.parameters()).device
    if new_tokens < 1 or warmup < 0 or repeats < 1:
        raise ValueError("new_tokens/repeats must be positive and warmup nonnegative")
    supported = [length for length in contexts if 1 <= length <= model.config.block_size]
    skipped = [length for length in contexts if length not in supported]
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    results = []
    autocast_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    for length in supported:
        prompt = torch.randint(model.config.vocab_size, (1, length), generator=generator,
                               device=device, dtype=torch.long)
        row = {"context_length": length, "precision": precision, "device": str(device),
               "skipped_contexts": skipped}
        for mode, use_cache in (("cache", True), ("no_cache", False)):
            for _ in range(warmup):
                with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    _run(model, prompt, new_tokens, use_cache, device)
            measurements = []
            for _ in range(repeats):
                with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    measurements.append(_run(model, prompt, new_tokens, use_cache, device))
            row[mode] = {
                key: statistics.median(item[key] for item in measurements)
                for key in measurements[0]
            }
        row["speedup_tokens_per_second"] = row["cache"]["tokens_per_second"] / max(
            row["no_cache"]["tokens_per_second"], 1e-12)
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="reports/benchmark_kv.json")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    results = benchmark(model, contexts=args.contexts, new_tokens=args.new_tokens,
                        warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                        precision=args.precision, device=device)
    report = {"checkpoint": args.checkpoint, "model_config": model.config.__dict__,
              "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
              "first_token_latency_definition": "prompt prefill plus first next-token prediction",
              "next_token_latency_definition": "decode-loop duration per subsequent generated token, excluding prefill and first token",
              "environment": environment_info(args.seed),
              "results": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["context_length", "precision", "device", "mode", "first_token_latency_ms",
                  "next_token_latency_ms", "tokens_per_second", "peak_vram_bytes",
                  "generated_tokens", "speedup_tokens_per_second"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for mode in ("cache", "no_cache"):
                writer.writerow({"context_length": row["context_length"],
                                 "precision": row["precision"], "device": row["device"],
                                 "mode": mode, **row[mode],
                                 "speedup_tokens_per_second": row["speedup_tokens_per_second"]})
    _write_svg(results, output.with_suffix(".svg"))
    print(json.dumps(report, indent=2))


def _write_svg(results: list[dict], path: Path) -> None:
    """Write a dependency-free throughput chart for cache/no-cache runs."""
    width, height = 760, 460
    left, right, top, bottom = 80, 30, 35, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [row[mode]["tokens_per_second"] for row in results for mode in ("cache", "no_cache")]
    ymax = max(values, default=1.0) * 1.1 or 1.0
    contexts = [row["context_length"] for row in results]
    x_positions = [left + plot_w * (index + 0.5) / max(1, len(contexts)) for index in range(len(contexts))]
    bar_w = min(60, plot_w / max(1, len(contexts)) / 3)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Decode throughput: KV cache vs no cache</text>']
    for tick in range(6):
        y_value = ymax * tick / 5
        y = top + plot_h - plot_h * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y_value:.1f}</text>')
    for index, row in enumerate(results):
        center = x_positions[index]
        for offset, mode, color in ((-bar_w/2, "cache", "#2474b5"), (bar_w/2, "no_cache", "#e07a2d")):
            value = row[mode]["tokens_per_second"]
            bar_h = plot_h * value / ymax
            x, y = center + offset - bar_w/2, top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["context_length"]}</text>')
    parts += [f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="13">Prompt context length (tokens)</text>',
              f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Generated tokens / second</text>',
              '<rect x="500" y="42" width="12" height="12" fill="#2474b5"/><text x="518" y="53" font-family="sans-serif" font-size="12">KV cache</text>',
              '<rect x="600" y="42" width="12" height="12" fill="#e07a2d"/><text x="618" y="53" font-family="sans-serif" font-size="12">No cache</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
