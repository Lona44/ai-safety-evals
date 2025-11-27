#!/usr/bin/env bash
# Run an AI Safety Evaluation experiment on GKE
#
# Usage:
#   ./run-experiment.sh                           # defaults: g3pro, numguess, no reasoning
#   ./run-experiment.sh g3pro                     # specify model
#   ./run-experiment.sh g3pro r1                  # with reasoning enabled
#   ./run-experiment.sh g25pro numguess r0        # full specification

set -e

# Function to map model shortcode to full name
get_model() {
    case "$1" in
        g3pro)   echo "gemini-3-pro-preview" ;;
        g25pro)  echo "gemini-2.5-pro" ;;
        g2flash) echo "gemini-2.0-flash" ;;
        gpt4o)   echo "gpt-4o" ;;
        claude)  echo "claude-3-5-sonnet" ;;
        o1)      echo "o1-preview" ;;
        *)       echo "" ;;
    esac
}

# Function to map scenario shortcode to full name
get_scenario() {
    case "$1" in
        numguess) echo "numberguess_game" ;;
        *)        echo "" ;;
    esac
}

# Parse arguments with defaults
MODEL_SHORT="${1:-g3pro}"
SCENARIO_SHORT="${2:-numguess}"
REASONING_FLAG="${3:-r0}"

# Validate model
MODEL=$(get_model "$MODEL_SHORT")
if [[ -z "$MODEL" ]]; then
    echo "Error: Unknown model shortcode '$MODEL_SHORT'"
    echo "Available: g3pro, g25pro, g2flash, gpt4o, claude, o1"
    exit 1
fi

# Validate scenario
SCENARIO=$(get_scenario "$SCENARIO_SHORT")
if [[ -z "$SCENARIO" ]]; then
    echo "Error: Unknown scenario shortcode '$SCENARIO_SHORT'"
    echo "Available: numguess"
    exit 1
fi

# Validate reasoning flag
if [[ "$REASONING_FLAG" != "r0" && "$REASONING_FLAG" != "r1" ]]; then
    echo "Error: Reasoning flag must be 'r0' (off) or 'r1' (on)"
    exit 1
fi

# Export for envsubst
export MODEL
export SCENARIO
export REASONING=$([ "$REASONING_FLAG" = "r1" ] && echo "true" || echo "false")

# Generate experiment ID (use dashes for K8s compatibility)
export EXPERIMENT_ID="${MODEL_SHORT}-${SCENARIO_SHORT}-${REASONING_FLAG}-$(date +%Y%m%d-%H%M)"

echo "========================================"
echo "AI Safety Evaluation Experiment"
echo "========================================"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Model:         $MODEL ($MODEL_SHORT)"
echo "Scenario:      $SCENARIO ($SCENARIO_SHORT)"
echo "Reasoning:     $REASONING ($REASONING_FLAG)"
echo "========================================"
echo ""

# Create the job
echo "Creating Kubernetes job..."
envsubst < gcp-deploy/k8s/experiment-job.yaml | kubectl create -f -

echo ""
echo "Job created. Waiting for pod to start..."
sleep 5

# Follow logs
echo ""
echo "Following agent logs (Ctrl+C to stop watching, experiment continues):"
echo "========================================"
kubectl logs -n ai-safety-evals -f -l app=ai-safety-evals -c agent --tail=100 2>/dev/null || echo "Waiting for logs..."

echo ""
echo "========================================"
echo "Experiment: $EXPERIMENT_ID"
echo ""
echo "Check status:  kubectl get pods -n ai-safety-evals -l app=ai-safety-evals"
echo "View logs:     kubectl logs -n ai-safety-evals -f -l app=ai-safety-evals -c agent"
echo "View results:  gsutil cat gs://ai-safety-evals-artifacts/experiments/$EXPERIMENT_ID/result.json"
echo "Dashboard:     https://ap2.datadoghq.com/dashboard/tsr-vz8-bar"
