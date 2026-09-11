from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from common import SPECIAL_TOKENS, iter_text_files, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.yaml")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="artifacts/tokenizer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok_cfg = cfg["tokenizer"]
    files = [str(p) for p in iter_text_files(args.data)]
    if not files:
        raise SystemExit(f"No .txt/.md files found in {args.data}")

    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=int(tok_cfg["vocab_size"]),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        special_tokens=list(SPECIAL_TOKENS.values()),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    tokenizer_json = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    (out / "vocab.json").write_text(
        json.dumps(tokenizer_json["model"]["vocab"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    merges = tokenizer_json["model"].get("merges", [])
    (out / "merges.txt").write_text(
        "\n".join(" ".join(pair) for pair in merges) + ("\n" if merges else ""),
        encoding="utf-8",
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=int(tok_cfg["model_max_length"]),
        **SPECIAL_TOKENS,
    )
    fast.init_kwargs["tokenizer_class"] = "GPT2TokenizerFast"
    fast.save_pretrained(out)
    tokenizer_config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_config["tokenizer_class"] = "GPT2TokenizerFast"
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2),
        encoding="utf-8",
    )

    print(f"Tokenizer written to {out}")
    print(f"Vocabulary size: {len(fast):,}")
    sample = "A small Llama-style model learns to predict the next token."
    print("Sample tokens:", fast.tokenize(sample))


if __name__ == "__main__":
    main()
