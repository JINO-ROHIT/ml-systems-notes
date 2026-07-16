# local dynamo setup experiments

steps to serve non kubernetes local path to run `Qwen/Qwen3-0.6B` with dynamo + sglang for 1 aggregated server

### update packages

```bash
apt-get update
apt-get install -y python3-dev libnuma1
```

## environment

```bash
cd /workspace/dynamo

uv venv /workspace/dynamo/.venv-dynamo-sglang --python 3.10
. /workspace/dynamo/.venv-dynamo-sglang/bin/activate

uv pip install pip
uv pip install --prerelease=allow "ai-dynamo[sglang]"
uv pip install "kernels>=0.12,<0.13"
```
 
(not really needed)

Use a local file discovery store and local Hugging Face cache:

```bash
mkdir -p /workspace/dynamo/local-run-logs
mkdir -p /workspace/dynamo/.dynamo-file-kv
mkdir -p /workspace/dynamo/.hf-cache

export DYN_FILE_KV=/workspace/dynamo/.dynamo-file-kv
export HF_HOME=/workspace/dynamo/.hf-cache
export CUDA_VISIBLE_DEVICES=0
```

for better rate limits

```bash
export HF_TOKEN=xxx
```

## dynamo frontend


```bash

export DYN_FILE_KV=/workspace/dynamo/.dynamo-file-kv
export HF_HOME=/workspace/dynamo/.hf-cache

python3 -m dynamo.frontend \
  --discovery-backend file \
  --http-port 8000
```

## sglang worker


```bash

export DYN_FILE_KV=/workspace/dynamo/.dynamo-file-kv
export HF_HOME=/workspace/dynamo/.hf-cache
export CUDA_VISIBLE_DEVICES=0

python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --discovery-backend file \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disable-cuda-graph \
  --mem-fraction-static 0.40 \
  --max-total-tokens 8192
```


## test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Write a one-sentence recipe for tea."}],
    "max_tokens": 64
  }'
```


