"""Train the from-scratch Dense GPT."""
import argparse
import json

from src.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--train-tokens")
    parser.add_argument("--validation-tokens")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or an indexed device such as cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--overfit-batch", action="store_true")
    args = parser.parse_args()
    summary = train(args.config, train_tokens_path=args.train_tokens,
                    validation_tokens_path=args.validation_tokens,
                    output_dir=args.output_dir, max_steps=args.max_steps,
                    device_name=args.device, resume=args.resume,
                    overfit_batch=args.overfit_batch)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
