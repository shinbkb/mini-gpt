"""Prepare bounded TinyStories token streams and train a custom byte-level BPE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from src.data.dataset import CausalLMDataset, iter_texts
from src.tokenizer.bpe import ByteLevelBPE


def _hash_texts(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--validation-limit", type=int, default=1_000)
    parser.add_argument("--vocab-size", type=int, default=8_000)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--revision", help="Hugging Face dataset commit hash")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--require-full-vocab", action="store_true")
    args = parser.parse_args()
    if bool(args.train_file) != bool(args.validation_file):
        parser.error("provide both --train-file and --validation-file, or neither")
    if args.train_file and args.train_file.resolve() == args.validation_file.resolve():
        parser.error("train and validation must use separate files")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")

    train_texts = list(iter_texts("train", args.train_file, args.train_limit, args.revision))
    validation_texts = list(iter_texts("validation", args.validation_file,
                                       args.validation_limit, args.revision))
    if not train_texts or not validation_texts:
        parser.error("train and validation must both contain at least one story")

    started = perf_counter()
    tokenizer = ByteLevelBPE().train(train_texts, vocab_size=args.vocab_size)
    train_seconds = perf_counter() - started
    if args.require_full_vocab and tokenizer.vocab_size != args.vocab_size:
        parser.error(f"corpus produced {tokenizer.vocab_size} tokens; requested {args.vocab_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    metrics = {
        "dataset": "local-files" if args.train_file else "roneneldan/TinyStories",
        "dataset_revision": args.revision,
        "sample_policy": "first N records, no shuffle",
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.vocab_size,
        "target_vocab_size": args.vocab_size,
        "tokenizer_train_seconds": train_seconds,
        "block_size": args.block_size,
        "splits": {},
    }
    for split, texts in (("train", train_texts), ("validation", validation_texts)):
        started = perf_counter()
        data = CausalLMDataset.from_texts(texts, tokenizer, args.block_size)
        encode_seconds = perf_counter() - started
        torch.save(data.tokens, args.output_dir / f"{split}_tokens.pt")
        metrics["splits"][split] = {
            "stories": len(texts), "tokens": data.tokens.numel(),
            "windows": len(data), "encode_seconds": encode_seconds,
            "tokens_per_second": data.tokens.numel() / max(encode_seconds, 1e-12),
            "sample_sha256": _hash_texts(texts),
        }
    (args.output_dir / "tokenizer_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
