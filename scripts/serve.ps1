param(
  [string]$Llama = $(if ($env:LLAMA_BIN) { $env:LLAMA_BIN } elseif ($env:LLAMA) { $env:LLAMA } else { "llama" }),
  [string]$Model = $(if ($env:LLAMA_MODEL) { $env:LLAMA_MODEL } elseif ($env:MODEL) { $env:MODEL } else { "artifacts/tiny-llama-f16.gguf" }),
  [string]$Host = $(if ($env:LLAMA_HOST) { $env:LLAMA_HOST } elseif ($env:HOST) { $env:HOST } else { "127.0.0.1" }),
  [int]$Port = $(if ($env:LLAMA_PORT) { [int]$env:LLAMA_PORT } elseif ($env:PORT) { [int]$env:PORT } else { 8080 }),
  [int]$ContextSize = $(if ($env:LLAMA_CONTEXT_SIZE) { [int]$env:LLAMA_CONTEXT_SIZE } elseif ($env:CONTEXT_SIZE) { [int]$env:CONTEXT_SIZE } else { 1024 })
)

if (-not (Get-Command $Llama -ErrorAction SilentlyContinue)) {
  throw "Could not find llama.cpp server binary: $Llama"
}

if (-not (Test-Path $Model)) {
  throw "Could not find GGUF model file: $Model"
}

Write-Host "Starting llama.cpp server on ${Host}:${Port} with model $Model"
& $Llama serve -m $Model --host $Host --port $Port -c $ContextSize
