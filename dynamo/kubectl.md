### k8 and kubectl

k8s is an open-source system for automating deployment, scaling, and management of containerized applications.

- **Node** - A single machine (physical or virtual) in the cluster
- **Pod** - The smallest deployable unit; a wrapper around one or more containers that share networking and storage
- **Deployment** - A controller that manages pod replicas, updates, and rollbacks
- **Service** - A stable network endpoint that exposes pods to internal or external traffic
- **Namespace** - A virtual cluster inside a cluster, used to isolate resources
- **Cluster** - A set of nodes managed together by Kubernetes
(check and improve definitions)


`kubectl` is the command-line tool used to interact with a Kubernetes cluster. Every command follows the pattern: `kubectl <action> <resource> [options]`

minikube is a tool that runs a single-node Kubernetes cluster locally for development and testing.


#### Cluster Info
```bash
kubectl cluster-info          # Shows master API server URL and cluster services
kubectl get nodes             # Lists all nodes in the cluster with their status
kubectl version               # Displays kubectl and Kubernetes server versions
```

#### Deployments
A Deployment manages a set of identical pods, ensuring the desired number are running and handles updates.

```bash
kubectl get deployments                              # List all deployments
kubectl create deployment nginx --image=nginx        # Create a deployment running nginx image
kubectl scale deployment nginx --replicas=3          # Scale up to 3 pod replicas
kubectl rollout status deployment nginx              # Watch the rollout progress
kubectl rollout undo deployment nginx                # Roll back to the previous version
```

#### Pods
A Pod wraps a container. Pods are ephemeral - they can be killed and replaced at any time.

```bash
kubectl get pods                                     # List pods in current namespace
kubectl get pods -o wide                             # List pods with extra details (node, IPs)
kubectl describe pod <pod-name>                      # Show detailed info and events for a pod
kubectl logs <pod-name>                              # View stdout/stderr logs from a pod
kubectl exec -it <pod-name> -- /bin/bash             # Open an interactive shell inside a running pod
```

#### Services
A Service provides a stable IP and DNS name to route traffic to a set of pods.

```bash
kubectl get svc                                      # List all services
kubectl expose deployment nginx --port=80 --type=NodePort   # Expose deployment on a node port
```

**Service Types:**
- **ClusterIP** (default) - Internal access only
- **NodePort** - Exposes on a static port on each node
- **LoadBalancer** - Provisions an external load balancer (cloud providers)

#### Apply/Delete (Declarative Management)
You can define resources in YAML files and let Kubernetes manage them declaratively.

```bash
kubectl apply -f <file.yaml>                         # Create or update resources from a YAML file
kubectl delete -f <file.yaml>                        # Delete resources defined in the YAML file
kubectl delete pod <pod-name>                        # Delete a specific pod
```

#### Namespaces
Namespaces let you divide cluster resources between teams or environments (dev, staging, prod).

```bash
kubectl get namespaces                               # List all namespaces
kubectl get pods -n <namespace>                      # List pods in a specific namespace
```

#### Minikube (Local Development)
Minikube creates a single-node cluster on your local machine.

```bash
minikube start                                       # Start a local Kubernetes cluster
minikube status                                      # Check if minikube is running
minikube dashboard                                   # Open the Kubernetes web dashboard
minikube service <service-name>                      # Open a service in the browser
```
