"""Generate text from a trained Dense GPT checkpoint."""
import argparse
import json
import torch

from src.models.gpt import GPT, GPTConfig
from src.tokenizer.bpe import ByteLevelBPE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["model_config"])
    tokenizer = ByteLevelBPE.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match checkpoint")
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = model.generate(input_ids, args.max_new_tokens, temperature=args.temperature,
                            top_k=args.top_k, eos_id=tokenizer.eos_id, generator=generator)
    print(tokenizer.decode(result[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
