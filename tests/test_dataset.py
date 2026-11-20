import json

import pytest
import torch

from src.data.dataset import CausalLMDataset, iter_texts
from src.tokenizer.bpe import ByteLevelBPE


def test_shift_dtype_shape_and_eos():
    tokenizer = ByteLevelBPE()
    dataset = CausalLMDataset.from_texts(["ab", "cd"], tokenizer, block_size=2)
    assert dataset.tokens.tolist() == [97, 98, 258, 99, 100, 258]
    assert len(dataset) == 2
    x, y = dataset[1]
    assert x.dtype == y.dtype == torch.long
    assert x.shape == y.shape == (2,)
    assert x.tolist() == [258, 99]
    assert y.tolist() == [99, 100]
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    assert len(CausalLMDataset([], block_size=4)) == 0


def test_local_jsonl_and_txt(tmp_path):
    path = tmp_path / "stories.jsonl"
    path.write_text("\n".join(json.dumps({"text": story})
                                for story in ["line one\nline two", "Tiếng Việt", "three"]),
                    encoding="utf-8")
    assert list(iter_texts("train", path, limit=2)) == ["line one\nline two", "Tiếng Việt"]
    path = tmp_path / "stories.txt"
    path.write_text("first\nstory\n<|endoftext|>\nsecond", encoding="utf-8")
    assert list(iter_texts("validation", path)) == ["first\nstory\n", "second"]


def test_invalid_data_inputs(tmp_path):
    with pytest.raises(ValueError):
        list(iter_texts("test"))
    with pytest.raises(ValueError):
        list(iter_texts("train", limit=0))
    with pytest.raises(FileNotFoundError):
        list(iter_texts("train", tmp_path / "missing.jsonl"))
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"text": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_texts("train", bad_json))
    with pytest.raises(ValueError):
        CausalLMDataset([1, -2], block_size=4)
