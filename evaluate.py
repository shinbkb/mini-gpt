"""Evaluate a Dense GPT checkpoint on an M1 token cache."""
import argparse
import json
from pathlib import Path

import torch

from src.models.gpt import GPT, GPTConfig
from src.training.trainer import evaluate_loss, load_tokens, _ppl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", default="artifacts/validation_tokens.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokens = load_tokens(args.tokens, config.vocab_size)
    loss = evaluate_loss(model, tokens, batch_size=args.batch_size,
                         eval_batches=args.batches, device=device, seed=987)
    print(json.dumps({"checkpoint": str(Path(args.checkpoint)), "tokens": args.tokens,
                      "loss": loss, "perplexity": _ppl(loss), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
