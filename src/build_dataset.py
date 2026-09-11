from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import load_config, load_corpus_text, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--out", default="artifacts/dataset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    split = float(cfg["training"].get("train_split", 0.98))
    text = load_corpus_text(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    ids = tok.encode(text, add_special_tokens=False)
    ids.append(tok.eos_token_id)
    dtype = np.uint16 if len(tok) < 65536 else np.uint32
    arr = np.asarray(ids, dtype=dtype)

    cut = max(1, min(len(arr) - 1, int(len(arr) * split)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arr[:cut].tofile(out / "train.bin")
    arr[cut:].tofile(out / "val.bin")
    save_json(out / "meta.json", {
        "dtype": str(np.dtype(dtype)),
        "tokens_total": int(len(arr)),
        "tokens_train": int(cut),
        "tokens_val": int(len(arr) - cut),
        "vocab_size": int(len(tok)),
    })
    print(f"Tokenized {len(arr):,} tokens -> {out}")


if __name__ == "__main__":
    main()
