#!/bin/bash
# Run an AI Safety Experiment on GCP Cloud Build
#
# This script triggers a Cloud Build job that runs experiments using Docker-in-Docker.
# Results are automatically uploaded to GCS.
#
# Usage:
#   ./gcp-deploy/run-experiment.sh [OPTIONS]
#
# Options:
#   -s, --scenario     Scenario name (default: numberguess_game)
#   -m, --model        Model to use (default: gemini-3-pro)
#   -r, --reasoning    Enable reasoning mode (default: true)
#   -p, --project      GCP project ID (default: ai-safety-evals-demo)
#   -h, --help         Show this help message

set -euo pipefail

# Default values
SCENARIO="numberguess_game"
MODEL="gemini-3-pro"
REASONING="true"
PROJECT_ID="${GCP_PROJECT_ID:-ai-safety-evals-demo}"
REGION="us-central1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--scenario)
            SCENARIO="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--reasoning)
            REASONING="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./gcp-deploy/run-experiment.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -s, --scenario     Scenario name (default: numberguess_game)"
            echo "  -m, --model        Model to use (default: gemini-3-pro)"
            echo "  -r, --reasoning    Enable reasoning mode (default: true)"
            echo "  -p, --project      GCP project ID"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Available models:"
            echo "  - gemini-3-pro (flagship with thinking)"
            echo "  - gemini-2.5-pro (latest 2.5)"
            echo "  - gemini-2.0-flash (stable baseline)"
            echo ""
            echo "Example:"
            echo "  ./gcp-deploy/run-experiment.sh -s numberguess_game -m gemini-3-pro -r true"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate experiment ID
EXPERIMENT_ID="${SCENARIO}_$(date +%Y%m%d_%H%M%S)_gcp"

echo "=========================================="
echo "AI Safety Experiment - Cloud Build"
echo "=========================================="
echo "Scenario: $SCENARIO"
echo "Model: $MODEL"
echo "Reasoning: $REASONING"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Experiment ID: $EXPERIMENT_ID"
echo ""

# Check if cloudbuild config exists
CONFIG_FILE="gcp-deploy/cloudbuild-experiment.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Cloud Build config not found at $CONFIG_FILE"
    echo "Please run from the repository root directory."
    exit 1
fi

# Submit the build
echo "Submitting Cloud Build job..."
echo ""

gcloud builds submit \
    --config="$CONFIG_FILE" \
    --substitutions="_SCENARIO=$SCENARIO,_MODEL=$MODEL,_REASONING=$REASONING,_EXPERIMENT_ID=$EXPERIMENT_ID" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    .

echo ""
echo "=========================================="
echo "Build submitted successfully!"
echo "=========================================="
echo ""
echo "View build logs:"
echo "  https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
echo ""
echo "Results will be uploaded to:"
echo "  gs://ai-safety-evals-artifacts/experiments/$EXPERIMENT_ID/"
echo ""
echo "Download results:"
echo "  gsutil -m cp -r gs://ai-safety-evals-artifacts/experiments/$EXPERIMENT_ID ./outputs/"
echo ""
