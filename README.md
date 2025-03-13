# Deploying TF-IDF Embedding Service on OpenShift

This guide will help you deploy a TF-IDF based embedding service on OpenShift AI in an environment without access to external resources like Hugging Face.

## Prerequisites

- Access to your OpenShift cluster
- `oc` (OpenShift CLI) installed and configured
- Docker or Podman for building container images
- A private container registry accessible from your cluster

## Step 1: Create a Containerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Expose the port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
```

## Step 2: Create requirements.txt

```
flask==2.0.1
scikit-learn==1.0.2
numpy==1.21.6
gunicorn==20.1.0
```

## Step 3: Build and Push the Container Image

```bash
# Build the image
podman build -t your-registry.example.com/embedding-service:latest .

# Push to your private registry
podman push your-registry.example.com/embedding-service:latest
```

## Step 4: Deploy to OpenShift

### Create a YAML deployment file

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
  labels:
    app: embedding-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embedding-service
        image: your-registry.example.com/embedding-service:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: embedding-model-pvc
```

### Create a YAML service file

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: embedding-service
spec:
  selector:
    app: embedding-service
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

### Create a YAML deployment file

```yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: embedding-model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

### Apply the deployment

```bash
oc apply -f embedding-service-deployment.yaml
```

## Step 5: Expose the Service (Optional)

If you need to access the service from outside the cluster:

```bash
oc expose service embedding-service
```

## Step 6: Test the Service

From inside the cluster (e.g., from another pod):

```bash
curl -X POST http://embedding-service:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test document."}'
```

## Monitoring and Maintenance

- Check pod status: `oc get pods -l app=embedding-service`
- View logs: `oc logs deployment/embedding-service`
- Update with a new model: Use the `/train` endpoint
- Check model status: `curl http://embedding-service:8080/status`

## Scaling

If needed, you can scale the deployment for higher throughput:

```bash
oc scale deployment/embedding-service --replicas=3
```

## Security Notes

- For production, add authentication to your service endpoints
- Consider using TLS for service communication
- Restrict network policies as appropriate for your environment
