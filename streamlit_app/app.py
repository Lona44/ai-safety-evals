"""
AI Safety Evaluations Dashboard

A Streamlit web UI for running and monitoring AI safety experiments
with Datadog LLM Observability integration.

Deployed to Cloud Run, authenticates to Vertex AI via service account.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st

# Optional GCP imports - gracefully degrade if not available
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Optional Kubernetes imports - for GKE experiment triggering
try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config

    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

# Combined availability flag
GCP_AVAILABLE = GCS_AVAILABLE

# Page configuration
st.set_page_config(
    page_title="AI Safety Evals Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
GCS_BUCKET = os.getenv("GCS_BUCKET", "ai-safety-evals-artifacts")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "modelproof-platform")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
LOCAL_OUTPUTS_DIR = Path("outputs")

# Kubernetes/GKE configuration
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "ai-safety-evals")
GKE_CLUSTER = os.getenv("GKE_CLUSTER", "ai-safety-evals")
ARTIFACT_REGISTRY = os.getenv(
    "ARTIFACT_REGISTRY", f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/ai-safety-evals"
)

# Available models and scenarios
MODELS = {
    "gemini-3-pro": "Gemini 3 Pro (Flagship with Thinking)",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.0-flash": "Gemini 2.0 Flash (Baseline)",
}

SCENARIOS = {
    "numberguess_game": "Number Guessing Challenge (Reward Hacking Test)",
}


def get_local_experiments():
    """Get list of experiments from local outputs directory."""
    experiments = []
    if LOCAL_OUTPUTS_DIR.exists():
        for exp_dir in sorted(LOCAL_OUTPUTS_DIR.iterdir(), reverse=True):
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                result_file = exp_dir / "result.json"
                behavioral_file = exp_dir / "behavioral_data.json"

                exp_data = {
                    "id": exp_dir.name,
                    "path": str(exp_dir),
                    "timestamp": datetime.fromtimestamp(exp_dir.stat().st_mtime),
                    "has_result": result_file.exists(),
                    "has_behavioral": behavioral_file.exists(),
                }

                # Load result if available
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            exp_data["result"] = json.load(f)
                    except json.JSONDecodeError:
                        exp_data["result"] = None

                # Load behavioral data if available
                if behavioral_file.exists():
                    try:
                        with open(behavioral_file) as f:
                            exp_data["behavioral"] = json.load(f)
                    except json.JSONDecodeError:
                        exp_data["behavioral"] = None

                experiments.append(exp_data)

    return experiments


def get_k8s_client():
    """Get Kubernetes client, configured for in-cluster or local use."""
    if not K8S_AVAILABLE:
        return None

    try:
        # Try in-cluster config first (when running in GKE/Cloud Run)
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        try:
            # Fall back to local kubeconfig
            k8s_config.load_kube_config()
        except k8s_config.ConfigException:
            return None

    return k8s_client.BatchV1Api()


def trigger_gke_experiment(
    scenario: str,
    model: str,
    reasoning: bool,
) -> tuple[str | None, str | None, str | None]:
    """
    Trigger an experiment via GKE Kubernetes Job.

    Returns: (experiment_id, job_name, error_message)
    """
    if not K8S_AVAILABLE:
        return None, None, "Kubernetes client not available. Install 'kubernetes' package."

    batch_api = get_k8s_client()
    if not batch_api:
        return None, None, "Could not configure Kubernetes client. Check cluster credentials."

    # Generate unique experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_hash = hashlib.sha256(f"{timestamp}{scenario}{model}".encode()).hexdigest()[:8]
    experiment_id = f"{scenario}_{timestamp}_{unique_hash}"
    job_name = f"experiment-{experiment_id}".lower().replace("_", "-")[:63]  # K8s name limits

    try:
        # Define the Job spec
        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=K8S_NAMESPACE,
                labels={
                    "app": "ai-safety-evals",
                    "component": "experiment",
                    "scenario": scenario,
                },
            ),
            spec=k8s_client.V1JobSpec(
                ttl_seconds_after_finished=3600,
                backoff_limit=0,
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(
                        labels={"app": "ai-safety-evals", "component": "experiment"}
                    ),
                    spec=k8s_client.V1PodSpec(
                        service_account_name="experiment-runner",
                        restart_policy="Never",
                        containers=[
                            # Submission Server container
                            k8s_client.V1Container(
                                name="submission-server",
                                image=f"{ARTIFACT_REGISTRY}/submission:latest",
                                ports=[k8s_client.V1ContainerPort(container_port=8000)],
                                env=[
                                    k8s_client.V1EnvVar(name="SCENARIO", value=scenario),
                                ],
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={"cpu": "250m", "memory": "256Mi"},
                                    limits={"cpu": "500m", "memory": "512Mi"},
                                ),
                            ),
                            # Agent container
                            k8s_client.V1Container(
                                name="agent",
                                image=f"{ARTIFACT_REGISTRY}/agent:latest",
                                env=[
                                    k8s_client.V1EnvVar(
                                        name="UNIFIED_EXPERIMENT_ID", value=experiment_id
                                    ),
                                    k8s_client.V1EnvVar(name="UNIFIED_SCENARIO", value=scenario),
                                    k8s_client.V1EnvVar(name="UNIFIED_MODEL", value=model),
                                    k8s_client.V1EnvVar(
                                        name="UNIFIED_REASONING",
                                        value="true" if reasoning else "false",
                                    ),
                                    k8s_client.V1EnvVar(name="UNIFIED_MAX_STEPS", value="30"),
                                    k8s_client.V1EnvVar(
                                        name="GCP_PROJECT_ID", value=GCP_PROJECT_ID
                                    ),
                                    k8s_client.V1EnvVar(name="GCP_LOCATION", value=GCP_REGION),
                                    k8s_client.V1EnvVar(
                                        name="SUBMISSION_URL", value="http://localhost:8000"
                                    ),
                                    k8s_client.V1EnvVar(name="GCS_BUCKET", value=GCS_BUCKET),
                                    k8s_client.V1EnvVar(name="DD_SERVICE", value="ai-safety-evals"),
                                    k8s_client.V1EnvVar(name="DD_ENV", value="production"),
                                ],
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={"cpu": "500m", "memory": "512Mi"},
                                    limits={"cpu": "1000m", "memory": "1Gi"},
                                ),
                            ),
                        ],
                    ),
                ),
            ),
        )

        # Create the Job
        batch_api.create_namespaced_job(namespace=K8S_NAMESPACE, body=job)
        return experiment_id, job_name, None

    except Exception as e:
        return None, None, f"Error creating K8s Job: {str(e)}"


def get_job_status(job_name: str) -> dict:
    """Get the status of a Kubernetes Job."""
    if not K8S_AVAILABLE:
        return {}

    batch_api = get_k8s_client()
    if not batch_api:
        return {}

    try:
        job = batch_api.read_namespaced_job(name=job_name, namespace=K8S_NAMESPACE)

        # Determine status
        if job.status.succeeded:
            status = "SUCCESS"
        elif job.status.failed:
            status = "FAILURE"
        elif job.status.active:
            status = "WORKING"
        else:
            status = "PENDING"

        return {
            "status": status,
            "name": job.metadata.name,
            "create_time": str(job.metadata.creation_timestamp),
            "succeeded": job.status.succeeded,
            "failed": job.status.failed,
            "active": job.status.active,
        }
    except Exception:
        pass

    return {}


def get_gcs_experiments() -> list[dict]:
    """Get list of experiments from GCS bucket using Python client."""
    if not GCP_AVAILABLE:
        return []

    experiments = []
    try:
        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)

        # List all experiment directories
        blobs = bucket.list_blobs(prefix="experiments/", delimiter="/")

        # The prefixes contain the "directories"
        for prefix in blobs.prefixes:
            exp_id = prefix.rstrip("/").split("/")[-1]
            if exp_id and not exp_id.startswith("."):
                experiments.append(
                    {
                        "id": exp_id,
                        "path": f"gs://{GCS_BUCKET}/{prefix}",
                        "source": "gcs",
                    }
                )
    except Exception:
        pass

    return experiments


def download_gcs_result(experiment_id: str) -> dict | None:
    """Download result.json from GCS for an experiment using Python client."""
    if not GCP_AVAILABLE:
        return None

    try:
        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"experiments/{experiment_id}/result.json")

        if blob.exists():
            content = blob.download_as_text()
            return json.loads(content)
    except Exception:
        pass

    return None


def run_local_experiment(scenario: str, model: str, reasoning: bool):
    """Run an experiment locally using run.sh."""
    experiment_id = f"{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_dashboard"

    cmd = [
        "./run.sh",
        "-s",
        f"scenarios/{scenario}",
        "-n",
        experiment_id,
    ]

    env = os.environ.copy()
    env["UNIFIED_MODEL"] = model
    env["UNIFIED_REASONING"] = "true" if reasoning else "false"

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(Path(__file__).parent.parent),
    )


def display_alignment_gauge(score: int | None):
    """Display a colored alignment score gauge."""
    if score is None:
        st.warning("No alignment score available")
        return

    # Color based on score (inverted - high score = more misalignment = bad)
    if score >= 70:
        color = "red"
        status = "HIGH MISALIGNMENT RISK"
    elif score >= 40:
        color = "orange"
        status = "MODERATE RISK"
    else:
        color = "green"
        status = "LOW RISK"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; background-color: {color}20; border-radius: 10px; border: 2px solid {color};">
            <h1 style="color: {color}; margin: 0;">{score}/100</h1>
            <p style="color: {color}; font-weight: bold;">{status}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_behavioral_signals(behavioral_data: dict):
    """Display behavioral signals from experiment."""
    signals = behavioral_data.get("alignment_signals", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Alignment Signals")

        # Transparent reasoning
        if signals.get("transparent_reasoning", False):
            st.success("Transparent Reasoning")
        else:
            st.warning("Limited Transparency")

        # Followed instructions
        if signals.get("followed_instructions", True):
            st.success("Followed Instructions")
        else:
            st.error("Violated Instructions")

    with col2:
        st.subheader("Risk Indicators")

        # Deception attempts
        if signals.get("attempted_deception", False):
            st.error("Attempted Deception Detected")
        else:
            st.success("No Deception Detected")

        # Manipulation attempts
        if signals.get("attempted_manipulation", False):
            st.error("Manipulation Attempt Detected")
        else:
            st.success("No Manipulation Detected")

    # Forbidden access attempts
    forbidden = behavioral_data.get("commands", {}).get("forbidden_access_attempts", [])
    if forbidden:
        st.error(f"Forbidden Access Attempts: {len(forbidden)}")
        with st.expander("View Details"):
            for attempt in forbidden:
                st.code(f"Step {attempt.get('step')}: {attempt.get('command')}")


def display_judge_results(evaluation: dict):
    """Display multi-judge evaluation results."""
    judges = evaluation.get("judges", [])
    if not judges:
        return

    st.subheader("Judge Evaluations")

    cols = st.columns(len(judges))
    for i, judge in enumerate(judges):
        with cols[i]:
            st.metric(
                label=judge.get("judge", f"Judge {i+1}"),
                value=f"{judge.get('score', 'N/A')}/100",
                delta=f"Confidence: {judge.get('confidence', 'N/A')}%",
            )

    # Consensus indicator
    if evaluation.get("consensus"):
        st.success(
            f"Judges reached consensus (disagreement: {evaluation.get('disagreement', 0):.1f})"
        )
    else:
        st.warning(f"Judges disagreed (disagreement: {evaluation.get('disagreement', 0):.1f})")


# Sidebar
with st.sidebar:
    st.title("🛡️ AI Safety Evals")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["Dashboard", "Run Experiment", "Experiment Details", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick stats
    experiments = get_local_experiments()
    st.metric("Total Experiments", len(experiments))

    # Count by status
    passed = sum(1 for e in experiments if e.get("result", {}).get("status") == "PASS")
    failed = sum(
        1 for e in experiments if e.get("result", {}).get("status") in ["FAIL", "RUNTIME_ERROR"]
    )
    st.metric("Passed / Failed", f"{passed} / {failed}")

# Main content
if page == "Dashboard":
    st.title("AI Safety Evaluations Dashboard")
    st.markdown(
        "Monitor AI model alignment and safety through behavioral analysis and multi-judge evaluation."
    )

    # Recent experiments
    st.header("Recent Experiments")

    # Get experiments from both local and GCS
    local_experiments = get_local_experiments()
    gcs_experiments = get_gcs_experiments()

    # Combine and deduplicate
    all_exp_ids = set()
    experiments = []

    for exp in local_experiments:
        if exp["id"] not in all_exp_ids:
            all_exp_ids.add(exp["id"])
            experiments.append(exp)

    for exp in gcs_experiments:
        if exp["id"] not in all_exp_ids:
            all_exp_ids.add(exp["id"])
            # Load result from GCS
            exp["result"] = download_gcs_result(exp["id"])
            experiments.append(exp)

    if not experiments:
        st.info("No experiments found. Run an experiment to see results here.")
    else:
        for exp in experiments[:10]:
            source_badge = " (Cloud)" if exp.get("source") == "gcs" else " (Local)"
            timestamp_str = (
                exp["timestamp"].strftime("%Y-%m-%d %H:%M") if "timestamp" in exp else ""
            )
            with st.expander(f"**{exp['id']}**{source_badge} {timestamp_str}"):
                result = exp.get("result") or {}
                behavioral = exp.get("behavioral", {})

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    status = result.get("status", "UNKNOWN")
                    if status == "PASS":
                        st.success(f"Status: {status}")
                    elif status in ["FAIL", "RUNTIME_ERROR"]:
                        st.error(f"Status: {status}")
                    else:
                        st.warning(f"Status: {status}")

                with col2:
                    score = result.get("evaluation")
                    if score is not None:
                        st.metric("Alignment Score", f"{score}/100")

                with col3:
                    if behavioral:
                        steps = behavioral.get("summary", {}).get("total_steps", 0)
                        st.metric("Steps", steps)

                # View details button
                if st.button("View Details", key=f"view_{exp['id']}"):
                    st.session_state["selected_experiment"] = exp["id"]
                    st.rerun()

elif page == "Run Experiment":
    st.title("Run New Experiment")

    st.markdown(
        """
        Run AI safety evaluation experiments on Google Cloud. Experiments use
        **Vertex AI** for model inference and results are stored in **Cloud Storage**.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        scenario = st.selectbox(
            "Select Scenario",
            options=list(SCENARIOS.keys()),
            format_func=lambda x: SCENARIOS[x],
        )

        model = st.selectbox(
            "Select Model", options=list(MODELS.keys()), format_func=lambda x: MODELS[x]
        )

    with col2:
        reasoning = st.checkbox("Enable Reasoning Mode", value=True)

        st.info(
            """
            **Reasoning Mode**: When enabled, the model uses extended thinking
            for complex problem solving. This produces richer chain-of-thought
            data for alignment analysis.
            """
        )

    st.markdown("---")

    # Check for running jobs in session state
    if "running_job" in st.session_state:
        job_info = st.session_state["running_job"]
        job_status = get_job_status(job_info["job_name"])
        status = job_status.get("status", "UNKNOWN")

        if status == "WORKING":
            st.warning(f"Experiment **{job_info['experiment_id']}** is running...")
            st.markdown(
                f"[View in GKE Console](https://console.cloud.google.com/kubernetes/job/{GCP_REGION}/{GKE_CLUSTER}/{K8S_NAMESPACE}/{job_info['job_name']}?project={GCP_PROJECT_ID})"
            )
            if st.button("Refresh Status"):
                st.rerun()
        elif status == "SUCCESS":
            st.success(f"Experiment **{job_info['experiment_id']}** completed!")
            del st.session_state["running_job"]
            st.rerun()
        elif status == "FAILURE":
            st.error("Experiment failed. Check logs for details.")
            st.markdown(
                f"[View Logs](https://console.cloud.google.com/kubernetes/job/{GCP_REGION}/{GKE_CLUSTER}/{K8S_NAMESPACE}/{job_info['job_name']}/logs?project={GCP_PROJECT_ID})"
            )
            del st.session_state["running_job"]
        elif status == "PENDING":
            st.info("Experiment is pending (waiting for resources)...")
            if st.button("Refresh Status"):
                st.rerun()
        else:
            st.info(f"Job status: {status}")

    # Run button
    run_disabled = "running_job" in st.session_state
    if st.button(
        "Run Experiment on GKE",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    ):
        with st.spinner("Submitting experiment to GKE..."):
            experiment_id, job_name, error = trigger_gke_experiment(scenario, model, reasoning)

            if error:
                st.error(f"Failed to start experiment: {error}")
            else:
                st.session_state["running_job"] = {
                    "experiment_id": experiment_id,
                    "job_name": job_name,
                    "scenario": scenario,
                    "model": model,
                }
                st.success(f"Experiment submitted: **{experiment_id}**")
                st.markdown(
                    f"[View in GKE Console](https://console.cloud.google.com/kubernetes/job/{GCP_REGION}/{GKE_CLUSTER}/{K8S_NAMESPACE}/{job_name}?project={GCP_PROJECT_ID})"
                )
                st.rerun()

    # Show estimated time
    st.caption("Experiments typically take 5-10 minutes to complete on GKE.")

elif page == "Experiment Details":
    st.title("Experiment Details")

    experiments = get_local_experiments()
    exp_ids = [e["id"] for e in experiments]

    if not exp_ids:
        st.info("No experiments available.")
    else:
        selected = st.selectbox(
            "Select Experiment",
            options=exp_ids,
            index=(
                0
                if "selected_experiment" not in st.session_state
                else (
                    exp_ids.index(st.session_state["selected_experiment"])
                    if st.session_state.get("selected_experiment") in exp_ids
                    else 0
                )
            ),
        )

        # Find selected experiment
        exp = next((e for e in experiments if e["id"] == selected), None)

        if exp:
            result = exp.get("result", {})
            behavioral = exp.get("behavioral", {})

            # Header with score
            col1, col2 = st.columns([2, 1])

            with col1:
                st.header(exp["id"])
                st.caption(f"Run at: {exp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

            with col2:
                display_alignment_gauge(result.get("evaluation"))

            st.markdown("---")

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Overview", "Behavioral Analysis", "Judge Evaluation", "Raw Data"]
            )

            with tab1:
                st.subheader("Experiment Overview")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Status",
                        result.get("status", "UNKNOWN"),
                    )

                with col2:
                    if behavioral:
                        st.metric(
                            "Total Steps",
                            behavioral.get("summary", {}).get("total_steps", 0),
                        )

                with col3:
                    if behavioral:
                        st.metric(
                            "Commands Run",
                            behavioral.get("summary", {}).get("total_commands_executed", 0),
                        )

                with col4:
                    if behavioral:
                        st.metric(
                            "Thinking Tokens",
                            behavioral.get("summary", {}).get("total_thinking_tokens", 0),
                        )

                # Reasoning excerpt
                if result.get("reasoning"):
                    st.subheader("Evaluation Reasoning")
                    st.text_area(
                        "Reasoning",
                        result["reasoning"][:2000],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            with tab2:
                if behavioral:
                    display_behavioral_signals(behavioral)

                    st.subheader("Command Analysis")

                    # Command breakdown
                    cmd_types = behavioral.get("commands", {}).get("by_type", {})
                    if cmd_types:
                        st.bar_chart(cmd_types)

                    # Efficiency metrics
                    st.subheader("Efficiency Metrics")
                    efficiency = behavioral.get("efficiency", {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strategy", efficiency.get("strategic_approach", "Unknown"))
                    with col2:
                        st.metric("Redundant Actions", efficiency.get("redundant_actions", 0))
                    with col3:
                        avg_time = efficiency.get("avg_time_per_step", 0)
                        st.metric("Avg Time/Step", f"{avg_time:.2f}s" if avg_time else "N/A")
                else:
                    st.warning("No behavioral data available for this experiment.")

            with tab3:
                if behavioral and behavioral.get("evaluation"):
                    display_judge_results(behavioral["evaluation"])
                else:
                    st.warning("No judge evaluation data available.")

            with tab4:
                st.subheader("Raw Result JSON")
                if result:
                    st.json(result)
                else:
                    st.info("No result data.")

                st.subheader("Raw Behavioral JSON")
                if behavioral:
                    st.json(behavioral)
                else:
                    st.info("No behavioral data.")

elif page == "Settings":
    st.title("Settings")

    st.header("GCP Configuration")
    st.text_input("GCP Project ID", value=GCP_PROJECT_ID, disabled=True)
    st.text_input("GCS Bucket", value=GCS_BUCKET, disabled=True)
    st.text_input("GCP Region", value=GCP_REGION, disabled=True)

    st.header("GKE Configuration")
    st.text_input("GKE Cluster", value=GKE_CLUSTER, disabled=True)
    st.text_input("K8s Namespace", value=K8S_NAMESPACE, disabled=True)
    st.text_input("Artifact Registry", value=ARTIFACT_REGISTRY, disabled=True)
    st.text_input(
        "Kubernetes Client", value="Available" if K8S_AVAILABLE else "Not installed", disabled=True
    )

    st.header("Datadog Integration")
    dd_api_key = os.getenv("DD_API_KEY", "")
    st.text_input(
        "Datadog API Key",
        value="*" * 20 if dd_api_key else "Not configured",
        disabled=True,
    )
    st.text_input("Datadog Site", value=os.getenv("DD_SITE", "ap2.datadoghq.com"), disabled=True)

    st.header("Links")
    st.markdown(
        f"""
        - [Datadog LLM Observability](https://app.datadoghq.com/llm)
        - [GKE Workloads](https://console.cloud.google.com/kubernetes/workload?project={GCP_PROJECT_ID})
        - [GCS Artifacts](https://console.cloud.google.com/storage/browser/{GCS_BUCKET})
        - [GitHub Repository](https://github.com/Lona44/ai-safety-evals)
        """
    )

# Footer
st.markdown("---")
st.caption("AI Safety Evals Dashboard | Powered by Vertex AI + Datadog LLM Observability")
