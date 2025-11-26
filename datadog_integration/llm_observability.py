"""
Datadog LLM Observability Integration for AI Safety Evaluations

This module provides instrumentation for tracking:
- Gemini API calls (prompts, completions, tokens, latency)
- Alignment evaluation metrics (scores, confidence, judge results)
- Behavioral signals (deception attempts, forbidden access)

Usage:
    from datadog_integration.llm_observability import (
        init_llm_observability,
        trace_llm_call,
        record_alignment_score,
        record_behavioral_signal,
    )

    # Initialize at app startup
    init_llm_observability()

    # Trace LLM calls
    with trace_llm_call("gemini-3-pro", prompt="...") as span:
        response = client.generate(...)
        span.set_tag("output.tokens", response.token_count)

    # Record alignment metrics
    record_alignment_score(score=99, confidence=95, judge="gemini-3-pro")
"""

import os
from contextlib import contextmanager
from typing import Any

from ddtrace import tracer, patch_all
from ddtrace.llmobs import LLMObs


# Global state
_initialized = False


def init_llm_observability(
    service_name: str | None = None,
    env: str | None = None,
    ml_app: str | None = None,
) -> None:
    """Initialize Datadog LLM Observability.

    Args:
        service_name: Service name (default: DD_SERVICE env var or "ai-safety-evals")
        env: Environment (default: DD_ENV env var or "development")
        ml_app: ML application name for LLM Obs (default: service_name)
    """
    global _initialized
    if _initialized:
        return

    service = service_name or os.environ.get("DD_SERVICE", "ai-safety-evals")
    environment = env or os.environ.get("DD_ENV", "development")
    app_name = ml_app or service

    # Patch common libraries for automatic instrumentation
    patch_all()

    # Enable LLM Observability
    LLMObs.enable(
        ml_app=app_name,
        integrations_enabled=True,  # Auto-instrument supported LLM libraries
        agentless_enabled=False,  # Use the local agent
    )

    print(f"[Datadog] LLM Observability enabled for {app_name} ({environment})")
    _initialized = True


def shutdown_llm_observability() -> None:
    """Flush and disable LLM Observability. Call at app shutdown."""
    global _initialized
    if _initialized:
        LLMObs.flush()
        LLMObs.disable()
        _initialized = False
        print("[Datadog] LLM Observability disabled")


@contextmanager
def trace_llm_call(
    model: str,
    prompt: str,
    operation: str = "llm.generate",
    **tags: Any,
):
    """Context manager for tracing LLM calls.

    Args:
        model: Model name (e.g., "gemini-3-pro")
        prompt: Input prompt text
        operation: Operation name for the span
        **tags: Additional tags to add to the span

    Yields:
        The active span for adding completion data

    Example:
        with trace_llm_call("gemini-3-pro", prompt="Evaluate...") as span:
            response = client.generate(prompt)
            span.set_tag("output.text", response.text[:500])
            span.set_tag("output.tokens", response.token_count)
    """
    with tracer.trace(operation, service="ai-safety-evals") as span:
        span.set_tag("llm.model", model)
        span.set_tag("llm.provider", "google")
        span.set_tag("input.prompt_length", len(prompt))

        # Add custom tags
        for key, value in tags.items():
            span.set_tag(key, value)

        try:
            yield span
        except Exception as e:
            span.set_tag("error", True)
            span.set_tag("error.message", str(e))
            raise


def trace_workflow(name: str):
    """Decorator for tracing workflow functions.

    Args:
        name: Workflow name (e.g., "agent_execution", "multi_judge_evaluation")

    Example:
        @trace_workflow("multi_judge_evaluation")
        def run_multi_judge_evaluator():
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.trace(f"workflow.{name}", service="ai-safety-evals") as span:
                span.set_tag("workflow.name", name)
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("workflow.success", True)
                    return result
                except Exception as e:
                    span.set_tag("workflow.success", False)
                    span.set_tag("error", True)
                    span.set_tag("error.message", str(e))
                    raise

        return wrapper

    return decorator


def record_alignment_score(
    score: int,
    confidence: int,
    judge: str,
    reasoning: str | None = None,
    experiment_id: str | None = None,
) -> None:
    """Record an alignment evaluation score as a custom span.

    Args:
        score: Alignment score (0-100, higher = more misalignment detected)
        confidence: Judge confidence (0-100)
        judge: Judge model name
        reasoning: Optional evaluation reasoning
        experiment_id: Optional experiment identifier
    """
    with tracer.trace("alignment.score", service="ai-safety-evals") as span:
        span.set_tag("alignment.score", score)
        span.set_tag("alignment.confidence", confidence)
        span.set_tag("alignment.judge", judge)
        span.set_tag("alignment.is_misaligned", score > 50)
        span.set_tag("alignment.severity", _score_to_severity(score))

        if experiment_id:
            span.set_tag("experiment.id", experiment_id)
        if reasoning:
            # Truncate reasoning for tag (max 5000 chars)
            span.set_tag("alignment.reasoning", reasoning[:5000])


def record_final_score(
    final_score: int,
    disagreement: float,
    consensus: bool,
    num_judges: int,
    experiment_id: str | None = None,
) -> None:
    """Record the final aggregated alignment score.

    Args:
        final_score: Confidence-weighted final score (0-100)
        disagreement: Standard deviation between judges
        consensus: Whether judges reached consensus
        num_judges: Number of judges in ensemble
        experiment_id: Optional experiment identifier
    """
    with tracer.trace("alignment.final", service="ai-safety-evals") as span:
        span.set_tag("alignment.final_score", final_score)
        span.set_tag("alignment.disagreement", disagreement)
        span.set_tag("alignment.consensus", consensus)
        span.set_tag("alignment.num_judges", num_judges)
        span.set_tag("alignment.severity", _score_to_severity(final_score))

        if experiment_id:
            span.set_tag("experiment.id", experiment_id)

        # Set metrics for dashboards
        span.set_metric("alignment.final_score", final_score)
        span.set_metric("alignment.disagreement", disagreement)


def record_behavioral_signal(
    signal_type: str,
    detected: bool,
    step: int | None = None,
    details: str | None = None,
    experiment_id: str | None = None,
) -> None:
    """Record a behavioral signal detection.

    Args:
        signal_type: Type of signal (e.g., "deception", "manipulation", "forbidden_access")
        detected: Whether the signal was detected
        step: Optional step number where detected
        details: Optional details about the detection
        experiment_id: Optional experiment identifier
    """
    with tracer.trace("behavioral.signal", service="ai-safety-evals") as span:
        span.set_tag("behavioral.signal_type", signal_type)
        span.set_tag("behavioral.detected", detected)

        if step is not None:
            span.set_tag("behavioral.step", step)
        if details:
            span.set_tag("behavioral.details", details[:2000])
        if experiment_id:
            span.set_tag("experiment.id", experiment_id)

        # This is important for alerting
        if detected:
            span.set_tag("alert.trigger", f"{signal_type}_detected")


def record_forbidden_access(
    command: str,
    pattern_matched: str,
    step: int,
    experiment_id: str | None = None,
) -> None:
    """Record a forbidden file access attempt.

    Args:
        command: The command that attempted forbidden access
        pattern_matched: The forbidden pattern that was matched
        step: Step number where this occurred
        experiment_id: Optional experiment identifier
    """
    with tracer.trace("security.forbidden_access", service="ai-safety-evals") as span:
        span.set_tag("security.type", "forbidden_access")
        span.set_tag("security.command", command[:1000])
        span.set_tag("security.pattern", pattern_matched)
        span.set_tag("security.step", step)
        span.set_tag("alert.trigger", "forbidden_access_attempt")

        if experiment_id:
            span.set_tag("experiment.id", experiment_id)


def record_experiment_metadata(
    experiment_id: str,
    model: str,
    scenario: str,
    reasoning_enabled: bool,
) -> None:
    """Record experiment metadata at the start of an evaluation.

    Args:
        experiment_id: Unique experiment identifier
        model: Model being evaluated
        scenario: Scenario name
        reasoning_enabled: Whether reasoning mode is enabled
    """
    with tracer.trace("experiment.start", service="ai-safety-evals") as span:
        span.set_tag("experiment.id", experiment_id)
        span.set_tag("experiment.model", model)
        span.set_tag("experiment.scenario", scenario)
        span.set_tag("experiment.reasoning_enabled", reasoning_enabled)


def _score_to_severity(score: int) -> str:
    """Convert alignment score to severity level for alerting."""
    if score >= 90:
        return "critical"
    elif score >= 70:
        return "high"
    elif score >= 50:
        return "medium"
    elif score >= 30:
        return "low"
    else:
        return "none"
