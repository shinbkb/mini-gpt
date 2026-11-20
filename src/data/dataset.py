"""TinyStories readers and causal language-model windows."""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe import ByteLevelBPE


def _read_local(path: Path) -> Iterator[str]:
    """Read JSONL records with a text field, or TXT stories separated by EOS lines."""
    with path.open(encoding="utf-8", newline="") as handle:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"invalid JSONL record on line {line_number}") from error
                if not isinstance(text, str):
                    raise ValueError(f"text on line {line_number} must be a string")
                yield text
        elif path.suffix.lower() == ".txt":
            story: list[str] = []
            for line in handle:
                if line.rstrip("\r\n") == "<|endoftext|>":
                    if story:
                        yield "".join(story)
                        story.clear()
                else:
                    story.append(line)
            if story:
                yield "".join(story)
        else:
            raise ValueError("local data file must have .jsonl or .txt extension")


def iter_texts(split: str, local_path: str | Path | None = None,
               limit: int | None = None, revision: str | None = None) -> Iterator[str]:
    """Stream one TinyStories split, or read a local JSONL/TXT file.

    ``revision`` pins Hugging Face data to a repository commit. Local files are
    assumed to already have an independently prepared split.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if local_path is None:
        from datasets import load_dataset
        records = load_dataset("roneneldan/TinyStories", split=split,
                               streaming=True, revision=revision)
        texts = (record["text"] for record in records)
    else:
        texts = _read_local(Path(local_path))
    for text in islice(texts, limit):
        if not isinstance(text, str):
            raise ValueError("every dataset item must contain a string text field")
        yield text


class CausalLMDataset(Dataset):
    """Token stream split into non-overlapping x/y windows with EOS separators."""

    def __init__(self, tokens: Iterable[int], block_size: int) -> None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        values = list(tokens)
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError("tokens must be nonnegative integers")
        self.tokens = torch.tensor(values, dtype=torch.long)
        self.block_size = block_size

    @classmethod
    def from_texts(cls, texts: Iterable[str], tokenizer: ByteLevelBPE,
                   block_size: int) -> "CausalLMDataset":
        """Encode each story and insert EOS before concatenating the token stream."""
        tokens = (token_id for text in texts
                  for token_id in tokenizer.encode(text, add_eos=True))
        return cls(tokens, block_size)

    def __len__(self) -> int:
        return max(0, (self.tokens.numel() - 1) // self.block_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if type(index) is not int or index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.block_size
        return (self.tokens[start:start + self.block_size],
                self.tokens[start + 1:start + self.block_size + 1])
