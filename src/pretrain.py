from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

from common import choose_amp_dtype, choose_device, count_parameters, load_config, perplexity, save_json, set_seed


def load_memmap(path: Path, dtype_name: str):
    return np.memmap(path, dtype=np.dtype(dtype_name), mode="r")


def batch_from(data, batch_size: int, block_size: int, device: torch.device):
    if len(data) <= block_size + 1:
        raise ValueError(f"Dataset has only {len(data)} tokens; block_size={block_size} is too large")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(np.array(data[i:i+block_size], dtype=np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(np.array(data[i+1:i+1+block_size], dtype=np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, val, batches, batch_size, block_size, device, amp_dtype):
    model.eval()
    losses = []
    ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if amp_dtype else nullcontext
    for _ in range(batches):
        x, y = batch_from(val, batch_size, block_size, device)
        with ctx():
            loss = model(input_ids=x, labels=y).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer")
    ap.add_argument("--dataset", default="artifacts/dataset")
    ap.add_argument("--out", default="artifacts/model")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    tr = cfg["training"]
    mc = cfg["model"]
    device = choose_device()
    amp_dtype = choose_amp_dtype(device, str(tr.get("dtype", "auto")))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    meta = json.loads(Path(args.dataset, "meta.json").read_text())
    train = load_memmap(Path(args.dataset, "train.bin"), meta["dtype"])
    val = load_memmap(Path(args.dataset, "val.bin"), meta["dtype"])

    model_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=int(mc["hidden_size"]),
        intermediate_size=int(mc["intermediate_size"]),
        num_hidden_layers=int(mc["num_hidden_layers"]),
        num_attention_heads=int(mc["num_attention_heads"]),
        num_key_value_heads=int(mc["num_key_value_heads"]),
        max_position_embeddings=int(mc["max_position_embeddings"]),
        rms_norm_eps=float(mc["rms_norm_eps"]),
        rope_theta=float(mc["rope_theta"]),
        attention_bias=bool(mc["attention_bias"]),
        mlp_bias=bool(mc["mlp_bias"]),
        tie_word_embeddings=bool(mc["tie_word_embeddings"]),
        hidden_act=str(mc["hidden_act"]),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(model_cfg).to(device)
    params = count_parameters(model)
    print(f"device={device} amp={amp_dtype} parameters={params:,}")

    max_steps = args.max_steps or int(tr["max_steps"])
    accum = int(tr["gradient_accumulation_steps"])
    batch_size = int(tr["batch_size"])
    block_size = int(tr["block_size"])
    optimizer = AdamW(model.parameters(), lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(tr["warmup_steps"]), max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    autocast_ctx = (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype)) if amp_dtype else nullcontext

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for step in range(1, max_steps + 1):
        running = 0.0
        for _ in range(accum):
            x, y = batch_from(train, batch_size, block_size, device)
            with autocast_ctx():
                loss = model(input_ids=x, labels=y).loss / accum
            scaler.scale(loss).backward()
            running += float(loss.detach().cpu())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start
            print(f"step={step:5d} train_loss={running:.4f} lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed:.1f}s")

        if step % int(tr["eval_interval"]) == 0 or step == max_steps:
            val_loss = evaluate(model, val, int(tr["eval_batches"]), batch_size, block_size, device, amp_dtype)
            row = {"step": step, "train_loss": running, "val_loss": val_loss, "perplexity": perplexity(val_loss)}
            history.append(row)
            print(f"EVAL step={step} val_loss={val_loss:.4f} ppl={row['perplexity']:.2f}")
            save_json(out / "metrics.json", {"parameters": params, "history": history})

        if step % int(tr["save_interval"]) == 0 or step == max_steps:
            model.save_pretrained(out, safe_serialization=True)
            tokenizer.save_pretrained(out)
            for filename in ("vocab.json", "merges.txt"):
                source = Path(args.tokenizer, filename)
                if source.exists():
                    (out / filename).write_bytes(source.read_bytes())

    print(f"Saved Hugging Face model to {out}")


if __name__ == "__main__":
    main()
