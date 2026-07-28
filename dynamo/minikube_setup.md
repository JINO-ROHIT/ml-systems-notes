# Local Qwen3-0.6B SGLang Setup

This guide sets up a local single-node Kubernetes environment with GPU support, installs the Dynamo platform, deploys `Qwen/Qwen3-0.6B` with SGLang, and verifies the OpenAI-compatible endpoint.


| Tool or component | Why it is needed |
|-------------------|------------------|
| `kubectl` | Applies Kubernetes resources and inspects pods, services, CRDs, and logs. |
| `helm` | Installs the Dynamo platform chart. |
| `minikube` | Creates a local Kubernetes cluster on this machine. |
| NVIDIA device plugin | Advertises the host GPU to Kubernetes as `nvidia.com/gpu`. Minikube enables this when started with `--gpus all`. |
| Dynamo platform | Installs the Dynamo operator, CRDs, and NATS control-plane pieces required to reconcile `DynamoGraphDeployment`. |
| `hf-token-secret` | Provides the Hugging Face token to model download and runtime pods. |
| `model-cache` PVC | Stores model files so pods can reuse the downloaded model. |

## Prerequisites


## 1. Install CLI Tools

Install `kubectl`:

```bash
curl -fsSLo /tmp/kubectl \
  "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -o root -g root -m 0755 /tmp/kubectl /usr/local/bin/kubectl
kubectl version --client=true
```

Install `minikube`:

```bash
curl -fsSLo /tmp/minikube-linux-amd64 \
  https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
install -o root -g root -m 0755 /tmp/minikube-linux-amd64 /usr/local/bin/minikube
minikube version
```

Install `helm`:

```bash
curl -fsSLo /tmp/helm.tar.gz https://get.helm.sh/helm-v3.19.3-linux-amd64.tar.gz
tar -C /tmp -xzf /tmp/helm.tar.gz
install -o root -g root -m 0755 /tmp/linux-amd64/helm /usr/local/bin/helm
helm version --short
```

## 2. Start Local Kubernetes

Start Minikube with Docker and GPU support:

```bash
minikube start \
  --driver docker \
  --container-runtime docker \
  --gpus all \
  --memory=24000mb \
  --cpus=8 \
  --force
```

`--force` is only needed when running Minikube as `root`.

Verify the cluster and GPU resource:

```bash
kubectl get nodes -o wide
kubectl wait --for=condition=Ready pod --all -n kube-system --timeout=300s
kubectl describe node minikube | grep -A12 -E 'Capacity:|Allocatable:|nvidia.com/gpu'
kubectl get storageclass
```

Expected storage class:

```text
standard (default)
```

Expected GPU capacity:

```text
nvidia.com/gpu: 1
```

## 3. Install Dynamo Platform

Install the matching published Dynamo platform chart:

```bash
curl -fsSLo /tmp/dynamo-platform-1.2.1.tgz \
  https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-1.2.1.tgz

helm install dynamo-platform /tmp/dynamo-platform-1.2.1.tgz \
  --namespace dynamo-system \
  --create-namespace \
  --wait \
  --timeout=10m
```

Verify:

```bash
kubectl get pods -n dynamo-system -o wide
kubectl get crd | grep dynamo
```

The operator and NATS pods should be `Running`.

## 4. Create Namespace And Secret

Use a fresh Hugging Face token. Do not reuse a token pasted into chat or logs.

```bash
export NAMESPACE=qwen3-sglang
export HF_TOKEN=<your-new-hf-token>

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 5. Create Model Cache And Download Model

The checked-in PVC uses a placeholder storage class. For Minikube, use `standard`:

```bash
cd /root/dynamo/recipes/qwen3-0.6b

sed 's/storageClassName: "your-storage-class-name"/storageClassName: "standard"/' \
  model-cache/model-cache.yaml | kubectl apply -n "${NAMESPACE}" -f -
```

Start the model download:

```bash
kubectl apply -f model-cache/model-download.yaml -n "${NAMESPACE}"

kubectl wait --for=condition=Complete job/model-download \
  -n "${NAMESPACE}" \
  --timeout=3600s
```

Check status:

```bash
kubectl get pvc,job,pod -n "${NAMESPACE}"
```

## 6. Deploy Qwen3-0.6B With SGLang

Apply the SGLang aggregated deployment:

```bash
kubectl apply -f sglang/agg/deploy.yaml -n "${NAMESPACE}"

kubectl wait --for=condition=Ready dynamographdeployment/qwen3-0-6b-sglang-agg \
  -n "${NAMESPACE}" \
  --timeout=900s
```

Check the deployment:

```bash
kubectl get dgd,dcd,pods,svc -n "${NAMESPACE}" -o wide
```

Expected:

```text
dynamographdeployment/qwen3-0-6b-sglang-agg   Ready=True
frontend pod                                  Running
SglangWorker pod                              Running
```

## 7. Run Requests

Start a port-forward:

```bash
kubectl port-forward svc/qwen3-0-6b-sglang-agg-frontend \
  8000:8000 \
  -n "${NAMESPACE}"
```

In another terminal, list models:

```bash
curl -sS http://localhost:8000/v1/models | python3 -m json.tool
```

Run a chat completion:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Write one sentence about Dynamo."}],
    "max_tokens": 64
  }' | python3 -m json.tool
```

## Useful Debug Commands

Check all workload resources:

```bash
kubectl get all,dgd,dcd,dwm,pvc -n "${NAMESPACE}"
```

Watch events:

```bash
kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp
```

Frontend logs:

```bash
kubectl logs -n "${NAMESPACE}" deploy/qwen3-0-6b-sglang-agg-frontend --tail=200
```

SGLang worker logs:

```bash
kubectl logs -n "${NAMESPACE}" deploy/qwen3-0-6b-sglang-agg-sglangworker-2c4d167f --tail=200
```

If the worker hash changes after editing the deployment, find the worker deployment name:

```bash
kubectl get deploy -n "${NAMESPACE}" | grep sglangworker
```

## Cleanup

Delete the Qwen3 deployment:

```bash
kubectl delete dynamographdeployment qwen3-0-6b-sglang-agg -n "${NAMESPACE}"
kubectl delete job model-download -n "${NAMESPACE}" --ignore-not-found
kubectl delete pvc model-cache -n "${NAMESPACE}"
kubectl delete namespace "${NAMESPACE}"
```

Remove Dynamo platform:

```bash
helm uninstall dynamo-platform -n dynamo-system
kubectl delete namespace dynamo-system
```

Stop Minikube:

```bash
minikube stop
```

Delete Minikube completely:

```bash
minikube delete
```
