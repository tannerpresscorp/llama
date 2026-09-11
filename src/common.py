from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

SPECIAL_TOKENS = {
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_amp_dtype(device: torch.device, requested: str = "auto"):
    if requested == "fp32":
        return None
    if device.type == "cuda":
        if requested in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "cpu" and requested == "bf16":
        return torch.bfloat16
    return None


def iter_text_files(data_dir: str | Path) -> Iterable[Path]:
    root = Path(data_dir)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            yield p


def load_corpus_text(data_dir: str | Path) -> str:
    chunks = []
    for path in iter_text_files(data_dir):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            chunks.append(text)
    if not chunks:
        raise FileNotFoundError(f"No .txt or .md corpus files found under {data_dir}")
    return "\n\n".join(chunks)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
