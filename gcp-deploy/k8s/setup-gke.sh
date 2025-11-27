#!/bin/bash
# One-time GKE setup for AI Safety Evals
#
# This script:
# 1. Creates GCP service account for experiments
# 2. Grants necessary IAM roles
# 3. Sets up Workload Identity binding
# 4. Creates K8s namespace and service account
# 5. Creates secrets for Datadog
#
# Prerequisites:
# - gcloud CLI authenticated
# - kubectl configured for the cluster
# - GKE cluster already created

set -e

PROJECT_ID="modelproof-platform"
REGION="us-central1"
CLUSTER_NAME="ai-safety-evals"
NAMESPACE="ai-safety-evals"
K8S_SA="experiment-runner"
GCP_SA="experiment-runner"

echo "=== AI Safety Evals GKE Setup ==="
echo "Project: $PROJECT_ID"
echo "Cluster: $CLUSTER_NAME"
echo ""

# Get cluster credentials
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME \
  --region=$REGION \
  --project=$PROJECT_ID

# Create GCP service account for experiments
echo "Creating GCP service account..."
gcloud iam service-accounts create $GCP_SA \
  --display-name="AI Safety Experiment Runner" \
  --project=$PROJECT_ID 2>/dev/null || echo "Service account already exists"

GCP_SA_EMAIL="$GCP_SA@$PROJECT_ID.iam.gserviceaccount.com"

# Grant IAM roles to the service account
echo "Granting IAM roles..."

# Vertex AI access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$GCP_SA_EMAIL" \
  --role="roles/aiplatform.user" \
  --quiet

# GCS access for results
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$GCP_SA_EMAIL" \
  --role="roles/storage.objectAdmin" \
  --quiet

# Allow K8s SA to impersonate GCP SA (Workload Identity)
echo "Setting up Workload Identity binding..."
gcloud iam service-accounts add-iam-policy-binding $GCP_SA_EMAIL \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:$PROJECT_ID.svc.id.goog[$NAMESPACE/$K8S_SA]" \
  --project=$PROJECT_ID

# Create K8s namespace
echo "Creating Kubernetes namespace..."
kubectl apply -f namespace.yaml

# Create K8s service account with Workload Identity annotation
echo "Creating Kubernetes service account..."
kubectl apply -f service-account.yaml

# Create Datadog secrets (optional)
echo ""
echo "To add Datadog API key, run:"
echo "  kubectl create secret generic datadog-secrets \\"
echo "    --from-literal=api-key=YOUR_DD_API_KEY \\"
echo "    -n $NAMESPACE"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run an experiment:"
echo "  export EXPERIMENT_ID=test_\$(date +%Y%m%d_%H%M%S)"
echo "  export SCENARIO=numberguess_game"
echo "  export MODEL=gemini-2.0-flash"
echo "  export REASONING=true"
echo "  envsubst < experiment-job.yaml | kubectl apply -f -"
echo ""
echo "To watch experiment:"
echo "  kubectl logs -f job/experiment-\$EXPERIMENT_ID -n $NAMESPACE"
