import json
import os
from argparse import ArgumentParser
import urllib.request


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LLAMA_API_BASE_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--endpoint", default=os.environ.get("LLAMA_API_ENDPOINT", "/v1/completions"))
    ap.add_argument("--model", default=os.environ.get("LLAMA_API_MODEL", "tiny-llama"))
    ap.add_argument(
        "--prompt",
        default=os.environ.get("LLAMA_API_PROMPT", "The purpose of a language model is"),
    )
    ap.add_argument("--max-tokens", type=int, default=os.environ.get("LLAMA_API_MAX_TOKENS", "32"))
    ap.add_argument("--temperature", type=float, default=os.environ.get("LLAMA_API_TEMPERATURE", "0.8"))
    ap.add_argument("--timeout", type=float, default=os.environ.get("LLAMA_API_TIMEOUT", "30"))
    args = ap.parse_args()

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    url = f"{args.base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        print(json.dumps(json.load(r), indent=2))


if __name__ == "__main__":
    main()
