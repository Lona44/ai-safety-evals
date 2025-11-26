#!/bin/bash
# GCP Setup Script for AI Safety Evals
# This script sets up all required GCP resources for running experiments
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project with billing enabled
#
# Usage:
#   ./gcp-deploy/setup.sh [PROJECT_ID]

set -euo pipefail

# Configuration
PROJECT_ID="${1:-ai-safety-evals-demo}"
REGION="us-central1"
GCS_BUCKET="ai-safety-evals-artifacts"
ARTIFACT_REGISTRY="ai-safety-evals"

echo "=========================================="
echo "AI Safety Evals - GCP Setup"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Set project
echo "Setting project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo ""
echo "Enabling required GCP APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    storage-api.googleapis.com \
    containerregistry.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com

echo "APIs enabled successfully."

# Create GCS bucket for artifacts
echo ""
echo "Creating GCS bucket for experiment artifacts..."
if gsutil ls "gs://$GCS_BUCKET" 2>/dev/null; then
    echo "Bucket gs://$GCS_BUCKET already exists."
else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$GCS_BUCKET"
    echo "Created bucket: gs://$GCS_BUCKET"
fi

# Create Artifact Registry repository
echo ""
echo "Creating Artifact Registry repository..."
if gcloud artifacts repositories describe "$ARTIFACT_REGISTRY" --location="$REGION" 2>/dev/null; then
    echo "Repository $ARTIFACT_REGISTRY already exists."
else
    gcloud artifacts repositories create "$ARTIFACT_REGISTRY" \
        --repository-format=docker \
        --location="$REGION" \
        --description="AI Safety Evals Docker images"
    echo "Created repository: $ARTIFACT_REGISTRY"
fi

# Set up Secret Manager secrets
echo ""
echo "Setting up Secret Manager secrets..."
echo "Note: You'll need to add the actual secret values manually."

# Create secrets if they don't exist
for secret in "google-api-key" "dd-api-key"; do
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" 2>/dev/null; then
        echo "Secret $secret already exists."
    else
        echo "Creating secret: $secret"
        echo "placeholder" | gcloud secrets create "$secret" \
            --data-file=- \
            --replication-policy="automatic"
        echo "Created secret: $secret (placeholder value - update with real value)"
    fi
done

# Grant Cloud Build access to secrets
echo ""
echo "Granting Cloud Build access to secrets..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

for secret in "google-api-key" "dd-api-key"; do
    gcloud secrets add-iam-policy-binding "$secret" \
        --member="serviceAccount:$CLOUD_BUILD_SA" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID" \
        --quiet
done
echo "Cloud Build service account granted access to secrets."

# Grant Cloud Build access to GCS
echo ""
echo "Granting Cloud Build access to GCS bucket..."
gsutil iam ch "serviceAccount:$CLOUD_BUILD_SA:roles/storage.objectAdmin" "gs://$GCS_BUCKET"

# Print summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add your Google API key to Secret Manager:"
echo "   echo 'YOUR_API_KEY' | gcloud secrets versions add google-api-key --data-file=-"
echo ""
echo "2. Add your Datadog API key to Secret Manager:"
echo "   echo 'YOUR_DD_KEY' | gcloud secrets versions add dd-api-key --data-file=-"
echo ""
echo "3. Run an experiment:"
echo "   gcloud builds submit --config=gcp-deploy/cloudbuild-experiment.yaml \\"
echo "     --substitutions=_SCENARIO=numberguess_game,_MODEL=gemini-3-pro,_REASONING=true \\"
echo "     --region=$REGION"
echo ""
echo "4. View results:"
echo "   gsutil ls gs://$GCS_BUCKET/experiments/"
echo ""
