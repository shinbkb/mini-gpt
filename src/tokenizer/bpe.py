"""A small, deterministic byte-level BPE tokenizer implemented from scratch."""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import re
from typing import Iterable

# Preserve every character while keeping common whitespace, word and punctuation
# runs apart, so merges do not cross these reversible boundaries.
_PRETOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]+", re.UNICODE)
_SPECIAL_TOKENS = {"<|pad|>": 256, "<|bos|>": 257, "<|eos|>": 258}
_BASE_VOCAB_SIZE = 259


def _pair_counts(tokens: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(tokens, tokens[1:]))


def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


class ByteLevelBPE:
    """Train and use a deterministic BPE over UTF-8 byte sequences."""

    def __init__(self) -> None:
        self.special_tokens = dict(_SPECIAL_TOKENS)
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        self.merges: list[tuple[int, int]] = []
        self.target_vocab_size = _BASE_VOCAB_SIZE

    @property
    def vocab_size(self) -> int:
        """Total size, including the three reserved special tokens."""
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eos_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    def train(self, texts: Iterable[str], vocab_size: int = 8000) -> "ByteLevelBPE":
        """Fit on text iterable; ties resolve by lexicographic pair of token IDs.

        Pair frequencies are weighted by repeated pre-token count. After each
        merge, only sequences that contained the selected pair are recounted.
        Small corpora may run out of pairs before reaching the requested size.
        """
        if type(vocab_size) is not int or vocab_size < _BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be an integer >= {_BASE_VOCAB_SIZE}")
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings, not a string")

        words: Counter[bytes] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("every training item must be a string")
            words.update(piece.encode("utf-8") for piece in _PRETOKEN_RE.findall(text))
        if not words:
            raise ValueError("training corpus must contain nonempty text")

        self.__init__()
        self.target_vocab_size = vocab_size
        sequences = [list(word) for word in words]
        weights = list(words.values())
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
        for sequence_index, tokens in enumerate(sequences):
            for pair, count in _pair_counts(tokens).items():
                pair_counts[pair] += count * weights[sequence_index]
                pair_owners[pair].add(sequence_index)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        while self.vocab_size < vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair) != -negative_count:
                continue  # stale heap entry

            new_id = self.vocab_size
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append(pair)
            changed_pairs: set[tuple[int, int]] = set()
            for sequence_index in sorted(pair_owners.get(pair, ())):
                before = _pair_counts(sequences[sequence_index])
                sequences[sequence_index] = _merge_pair(sequences[sequence_index], pair, new_id)
                after = _pair_counts(sequences[sequence_index])
                for affected in before.keys() | after.keys():
                    delta = (after[affected] - before[affected]) * weights[sequence_index]
                    if delta:
                        pair_counts[affected] += delta
                        changed_pairs.add(affected)
                    if after[affected]:
                        pair_owners[affected].add(sequence_index)
                    else:
                        pair_owners[affected].discard(sequence_index)
            for affected in changed_pairs:
                if pair_counts[affected] > 0:
                    heapq.heappush(heap, (-pair_counts[affected], affected))
                else:
                    pair_counts.pop(affected, None)
                    pair_owners.pop(affected, None)
            if len(heap) > 4 * max(len(pair_counts), 1):
                heap = [(-count, item) for item, count in pair_counts.items()]
                heapq.heapify(heap)
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode literal text; special-token spellings are treated as text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        tokens: list[int] = [self.special_tokens["<|bos|>"]] if add_bos else []
        for piece in _PRETOKEN_RE.findall(text):
            byte_tokens = list(piece.encode("utf-8"))
            while len(byte_tokens) > 1:
                candidates = [(merge_ranks[pair], pair)
                              for pair in zip(byte_tokens, byte_tokens[1:]) if pair in merge_ranks]
                if not candidates:
                    break
                _, pair = min(candidates)
                byte_tokens = _merge_pair(byte_tokens, pair, _BASE_VOCAB_SIZE + merge_ranks[pair])
            tokens.extend(byte_tokens)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text, replacing malformed UTF-8 if necessary."""
        special_bytes = {idx: spelling.encode("utf-8") for spelling, idx in self.special_tokens.items()}
        output: list[bytes] = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"invalid token ID: {token_id!r}")
            if token_id in special_bytes:
                if not skip_special_tokens:
                    output.append(special_bytes[token_id])
            else:
                output.append(self.vocab[token_id])
        return b"".join(output).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Write versioned JSON; token bytes are stored as base64 strings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": 1,
            "pretokenizer": "unicode-runs-v1",
            "config": {"vocab_size": self.target_vocab_size},
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": {str(idx): base64.b64encode(value).decode("ascii")
                      for idx, value in self.vocab.items()},
        }
        path.write_text(json.dumps(document, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPE":
        """Load and validate JSON format, merge dependencies, and token bytes."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            if raw["version"] != 1 or raw["pretokenizer"] != "unicode-runs-v1":
                raise ValueError("unsupported tokenizer format")
            if raw["special_tokens"] != _SPECIAL_TOKENS:
                raise ValueError("invalid special tokens")
            tokenizer = cls()
            tokenizer.target_vocab_size = raw["config"]["vocab_size"]
            if type(tokenizer.target_vocab_size) is not int or tokenizer.target_vocab_size < _BASE_VOCAB_SIZE:
                raise ValueError("invalid target vocabulary size")
            for pair in raw["merges"]:
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(idx) is not int or idx not in tokenizer.vocab for idx in pair)):
                    raise ValueError("invalid merge dependency")
                if tuple(pair) in tokenizer.merges:
                    raise ValueError("duplicate merge")
                token_id = tokenizer.vocab_size
                tokenizer.vocab[token_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
                tokenizer.merges.append(tuple(pair))
            expected_vocab = {str(idx): base64.b64encode(value).decode("ascii")
                              for idx, value in tokenizer.vocab.items()}
            if raw["vocab"] != expected_vocab or tokenizer.vocab_size > tokenizer.target_vocab_size:
                raise ValueError("vocabulary does not match merge rules/config")
            return tokenizer
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("malformed tokenizer JSON") from error
