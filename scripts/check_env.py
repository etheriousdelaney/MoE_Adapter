#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_MODULES = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "lightning": "lightning",
    "transformers": "transformers",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "soundfile": "soundfile",
    "librosa": "librosa",
}

OPTIONAL_MODULES = {
    "flash_attn": "flash_attn",
    "wandb": "wandb",
}


def check_module(import_name: str, package_name: str, required: bool) -> bool:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        level = "ERROR" if required else "WARN"
        print(f"[{level}] {package_name}: import failed: {exc}")
        return not required

    version = getattr(module, "__version__", "unknown")
    print(f"[OK] {package_name}: {version}")
    return True


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")

    ok = True
    for import_name, package_name in REQUIRED_MODULES.items():
        ok = check_module(import_name, package_name, required=True) and ok
    for import_name, package_name in OPTIONAL_MODULES.items():
        ok = check_module(import_name, package_name, required=False) and ok

    try:
        import torch
    except Exception:
        return 1

    print(f"torch cuda build: {torch.version.cuda}")
    cuda_ok = torch.cuda.is_available()
    print(f"cuda available: {cuda_ok}")
    if not cuda_ok:
        print("[ERROR] CUDA is not available to PyTorch.")
        ok = False
    else:
        count = torch.cuda.device_count()
        print(f"cuda device count: {count}")
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            print(f"cuda device {idx}: {name}, capability={capability[0]}.{capability[1]}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
