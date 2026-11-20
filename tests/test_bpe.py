from collections import Counter
import json
import random
import re

import pytest

from src.tokenizer.bpe import ByteLevelBPE


@pytest.mark.parametrize("text", ["", "hello, world!", "Tiếng Việt có dấu", "a\n\tb\r\n", "🙂🚀", "<|eos|>", "e\u0301 é", "\x00\xff"])
def test_utf8_round_trip(text):
    tokenizer = ByteLevelBPE().train(["hello world Tiếng Việt 🙂", text], 300)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode(text, True, True), skip_special_tokens=True) == text


def test_toy_merge_and_tie_break():
    tokenizer = ByteLevelBPE().train(["aaaa"], 260)
    assert tokenizer.merges == [(97, 97)]
    assert tokenizer.encode("aaaa") == [259, 259]
    assert ByteLevelBPE().train(["ab ac"], 260).merges == [(97, 98)]


def test_determinism_and_save_load(tmp_path):
    texts = ["banana bandana", "Tiếng Việt 🍌"]
    first = ByteLevelBPE().train(texts, 290)
    second = ByteLevelBPE().train(reversed(texts), 290)
    assert first.merges == second.merges
    path = tmp_path / "nested" / "tokenizer.json"
    first.save(path)
    restored = ByteLevelBPE.load(path)
    assert restored.merges == first.merges
    for text in texts:
        assert restored.encode(text) == first.encode(text)
        assert restored.decode(restored.encode(text)) == text
    assert first.vocab_size <= 290
    assert len(set(first.special_tokens.values())) == 3
    assert first.decode([256, 257, 258]) == "<|pad|><|bos|><|eos|>"


def test_validation_and_bad_inputs(tmp_path):
    for size in (0, 258, 259.5, True):
        with pytest.raises(ValueError):
            ByteLevelBPE().train(["abc"], size)
    with pytest.raises(ValueError):
        ByteLevelBPE().train([], 300)
    with pytest.raises(TypeError):
        ByteLevelBPE().train("abc", 300)
    with pytest.raises(FileNotFoundError):
        ByteLevelBPE.load(tmp_path / "missing.json")
    path = tmp_path / "corrupt.json"
    ByteLevelBPE().train(["abc"], 260).save(path)
    raw = json.loads(path.read_text())
    raw["merges"][0] = [999, 999]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        ByteLevelBPE.load(path)
    for invalid in ([-1], [259], [True], [1.5]):
        with pytest.raises(ValueError):
            ByteLevelBPE().decode(invalid)


def _reference_merges(texts, target):
    """Slow independent implementation that recounts every pair each round."""
    sequences = [list(piece.encode()) for text in texts
                 for piece in re.findall(r"\s+|\w+|[^\w\s]+", text, re.UNICODE)]
    merges = []
    while 259 + len(merges) < target:
        counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        new_id = 259 + len(merges)
        result = []
        for seq in sequences:
            merged, index = [], 0
            while index < len(seq):
                if index + 1 < len(seq) and tuple(seq[index:index + 2]) == pair:
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(seq[index])
                    index += 1
            result.append(merged)
        sequences = result
        merges.append(pair)
    return merges


def test_incremental_counts_match_reference():
    rng = random.Random(42)
    for _ in range(20):
        texts = ["".join(rng.choices("aaaabbbc đ🙂", k=45)) for _ in range(6)]
        tokenizer = ByteLevelBPE().train(texts, 285)
        assert tokenizer.merges == _reference_merges(texts, 285)
        assert all(tokenizer.decode(tokenizer.encode(text)) == text for text in texts)
