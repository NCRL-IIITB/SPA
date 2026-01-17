#!/bin/bash
set -e

echo "Deploying FULL-ACCESS Ubuntu stress pod to Minikube..."

# Create privileged Ubuntu pod in Minikube
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ubuntu-stress
  labels:
    app: ubuntu-stress
spec:
  restartPolicy: Always
  hostPID: true
  hostNetwork: true
  containers:
  - name: ubuntu
    image: ubuntu:22.04
    command: ["/bin/bash", "-c", "--"]
    args: ["while true; do sleep 30; done;"]
    securityContext:
      runAsUser: 0
      privileged: true
      allowPrivilegeEscalation: true
      capabilities:
        add: ["SYS_ADMIN", "SYS_RESOURCE", "NET_ADMIN", "NET_RAW"]
    tty: true
    stdin: true
EOF

# Wait for pod to be ready
echo "Waiting for ubuntu-stress pod to be ready..."
kubectl wait --for=condition=Ready pod/ubuntu-stress --timeout=180s

# Install debugging and stress tools inside container
echo "Installing necessary tools inside container..."
kubectl exec -it ubuntu-stress -- bash -c "apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y stress-ng htop curl wget iputils-ping net-tools vim python3 nano && \
  echo 'Tools installed successfully!'"

echo "Ubuntu stress pod is ready and running in background."
echo ""
echo "To access the container shell anytime, run:"
echo "   kubectl exec -it ubuntu-stress -- bash"
echo ""
echo "To monitor pod status:"
echo "   kubectl get pods"
echo ""
echo "To delete it later:"
echo "   kubectl delete pod ubuntu-stress"

