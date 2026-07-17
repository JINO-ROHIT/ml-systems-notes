# issues when setting up dynamo locally


## incompatible `kernels` version

The resolver installed `kernels==0.16.0`, while `transformers==5.6.0`
declares compatibility with `kernels>=0.12,<0.13`.

Fix:

```bash
. /workspace/dynamo/.venv-dynamo-sglang/bin/activate
uv pip install "kernels>=0.12,<0.13"
```

## missing `libnuma.so.1`

Fix:

```bash
apt-get update
apt-get install -y libnuma1
```