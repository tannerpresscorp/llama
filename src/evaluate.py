from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import choose_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--dataset", default="artifacts/dataset")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()

    device = choose_device()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    meta = json.loads(Path(args.dataset, "meta.json").read_text())
    ids = np.fromfile(Path(args.dataset, "val.bin"), dtype=np.dtype(meta["dtype"])).astype(np.int64)
    input_ids = torch.from_numpy(ids).unsqueeze(0).to(device)

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for start in range(0, max(1, input_ids.size(1) - 1), args.stride):
            end = min(start + args.block_size, input_ids.size(1))
            chunk = input_ids[:, start:end]
            if chunk.size(1) < 2:
                break
            loss = model(input_ids=chunk, labels=chunk).loss
            predicted = chunk.size(1) - 1
            nll_sum += float(loss) * predicted
            n_tokens += predicted
            if end == input_ids.size(1):
                break

    mean_loss = nll_sum / max(1, n_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    print(f"validation_tokens={n_tokens:,}")
    print(f"cross_entropy={mean_loss:.6f}")
    print(f"perplexity={ppl:.3f}")

    prompt = "The purpose of a language model is"
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8, top_p=0.95)
    print("\nSAMPLE\n------")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
