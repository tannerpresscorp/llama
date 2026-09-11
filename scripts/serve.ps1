param(
  [string]$Llama = "llama",
  [string]$Model = "artifacts/tiny-llama-f16.gguf",
  [int]$Port = 8080
)
& $Llama serve -m $Model --host 127.0.0.1 --port $Port -c 1024
