import json
import urllib.request

payload = {
    "model": "tiny-llama",
    "prompt": "The purpose of a language model is",
    "max_tokens": 32,
    "temperature": 0.8,
}
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    print(json.dumps(json.load(r), indent=2))
