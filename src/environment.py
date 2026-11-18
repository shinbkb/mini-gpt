"""Report runtime capabilities and initialize reproducible random seeds."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import random

import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python and PyTorch; this does not guarantee deterministic GPU kernels."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info(seed: int = 42) -> dict:
    """Initialize seeds and report versions, device names and total VRAM."""
    seed_everything(seed)
    libraries = {}
    for name in ("torch", "pytest", "PyYAML"):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name,
                            "total_vram_bytes": props.total_memory})
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": libraries, "seed": seed, "device": "cuda" if available else "cpu",
            "cuda_available": available, "cuda_build_version": torch.version.cuda,
            "gpu_devices": devices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(environment_info(args.seed), indent=2))


if __name__ == "__main__":
    main()
