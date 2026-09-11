# Tiny Llama From Scratch

An end-to-end educational project that trains a small Llama-style causal language model from random weights using a tokenizer you train yourself.

## Pipeline

1. Prepare a plain-text corpus.
2. Train a byte-level BPE tokenizer.
3. Tokenize and split the corpus into train/validation binary files.
4. Instantiate `LlamaForCausalLM` from a fresh `LlamaConfig` (random weights).
5. Pretrain with next-token prediction.
6. Measure validation cross-entropy/perplexity and generate a sample.
7. Save a standard Hugging Face Llama checkpoint.
8. Convert it with `llama.cpp/convert_hf_to_gguf.py`.
9. Serve the GGUF model through current `llama serve` local OpenAI-compatible API.

## Default tiny architecture

- 6 transformer layers
- hidden size 384
- 6 attention heads
- 2 KV heads (GQA)
- SwiGLU-style Llama MLP via `hidden_act: silu`
- RMSNorm
- RoPE
- 8,192-token custom BPE vocabulary
- 1,024 maximum positions
- 512-token training blocks

The exact parameter count is printed when training starts.

## 1. Environment

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA CUDA, install the PyTorch build appropriate for your CUDA environment from pytorch.org before installing the remaining requirements if necessary.

The conversion helper also installs `sentencepiece` and registers this project's custom byte-level BPE tokenizer with the current llama.cpp converter. The llama.cpp checkout remains ignored because it is an external build dependency.

## 2. Add training data

Place UTF-8 `.txt` or `.md` files under:

```text
data/raw/
```

Do not train on material you do not have the right to use. Keep evaluation material out of the training corpus.

## 3. Train the tokenizer

```powershell
python src/train_tokenizer.py
```

This creates `artifacts/tokenizer/`.

## 4. Build train/validation token files

```powershell
python src/build_dataset.py
```

This creates `artifacts/dataset/train.bin`, `val.bin`, and `meta.json`.

## 5. Pretrain from random weights

Do a smoke test first:

```powershell
python src/pretrain.py --max-steps 10
```

Then run the configured training schedule:

```powershell
python src/pretrain.py
```

The checkpoint is saved in standard Hugging Face form under `artifacts/model/`.

## 6. Evaluate

```powershell
python src/evaluate.py
```

Primary metric: validation perplexity. Also inspect generated samples. For serious work, add held-out domain tests and benchmark contamination checks.

## 7. Convert to GGUF

Clone current llama.cpp separately:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
pip install -r .\llama.cpp\requirements.txt
```

Convert:

```powershell
python scripts/convert_to_gguf.py --llama-cpp .\llama.cpp
```

The default output is:

```text
artifacts/tiny-llama-f16.gguf
```

For this very small educational model, F16 is fine. Quantization is more useful after scaling the model up.

## 8. Run the local API

Install/build a current llama.cpp binary, then:

```powershell
.\scripts\serve.ps1 -Llama "C:\path\to\llama.exe"
```

Default server:

```text
http://127.0.0.1:8080
```

Test the OpenAI-compatible completions endpoint:

```powershell
python scripts/test_api.py
```

Or POST to:

```text
http://127.0.0.1:8080/v1/completions
```

## Scaling presets

Start with the default. Once the pipeline is proven, scale one variable at a time.

| Tier | Layers | Hidden | Heads | KV heads | Suggested use |
|---|---:|---:|---:|---:|---|
| Smoke | 4 | 256 | 4 | 2 | Validate pipeline |
| Tiny (default) | 6 | 384 | 6 | 2 | Learn pretraining |
| Small | 12 | 768 | 12 | 4 | Serious workstation experiment |
| Larger | 16+ | 1024+ | 16+ | 4+ | GPU-server territory |

Keep `hidden_size` divisible by `num_attention_heads`.

## What success looks like

A successful first run does **not** mean ChatGPT-like answers. It means:

- tokenizer trains and round-trips text correctly;
- training loss decreases;
- held-out validation loss/perplexity improves;
- the saved HF checkpoint reloads;
- `convert_hf_to_gguf.py` accepts the model;
- llama.cpp loads the GGUF;
- the API produces next-token completions.

That proves every major piece of an LLM system from data to local inference.
