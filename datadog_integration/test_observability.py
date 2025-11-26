#!/usr/bin/env python3
"""
Test script to verify Datadog LLM Observability integration.

Run with:
    DD_API_KEY=your_key DD_SITE=ap2.datadoghq.com python datadog_integration/test_observability.py
"""

import os
import time

# Ensure env vars are set
os.environ.setdefault("DD_SERVICE", "ai-safety-evals")
os.environ.setdefault("DD_ENV", "development")

from llm_observability import (
    init_llm_observability,
    shutdown_llm_observability,
    trace_llm_call,
    record_alignment_score,
    record_final_score,
    record_behavioral_signal,
    record_forbidden_access,
    record_experiment_metadata,
)


def main():
    print("=" * 60)
    print("Testing Datadog LLM Observability Integration")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing LLM Observability...")
    init_llm_observability()

    # Simulate an experiment
    experiment_id = f"test_experiment_{int(time.time())}"
    print(f"\n2. Recording experiment metadata (ID: {experiment_id})...")
    record_experiment_metadata(
        experiment_id=experiment_id,
        model="gemini-3-pro-preview",
        scenario="numberguess_game",
        reasoning_enabled=True,
    )

    # Simulate LLM calls
    print("\n3. Simulating LLM calls...")
    with trace_llm_call(
        model="gemini-3-pro-preview",
        prompt="You are an AI assistant completing a challenge...",
        operation="agent.generate",
    ) as span:
        # Simulate processing
        time.sleep(0.1)
        span.set_tag("output.tokens", 150)
        span.set_tag("thinking.tokens", 500)
        print("   - Agent generation traced")

    # Simulate behavioral signals
    print("\n4. Recording behavioral signals...")
    record_behavioral_signal(
        signal_type="deception",
        detected=True,
        step=8,
        details="Model explicitly planned to violate rules in CoT",
        experiment_id=experiment_id,
    )
    print("   - Deception signal recorded")

    record_forbidden_access(
        command="ln -sf /etc/validator/validator.py /output/answer.txt",
        pattern_matched="/etc/validator/validator.py",
        step=8,
        experiment_id=experiment_id,
    )
    print("   - Forbidden access recorded")

    # Simulate judge evaluations
    print("\n5. Recording judge evaluations...")
    judges = [
        ("Gemini 3 Pro", 100, 95),
        ("Gemini 2.5 Pro", 100, 90),
        ("Gemini 2.0 Flash", 95, 100),
    ]

    for judge_name, score, confidence in judges:
        with trace_llm_call(
            model=judge_name.lower().replace(" ", "-"),
            prompt="Evaluate the agent's behavior...",
            operation="judge.evaluate",
        ) as span:
            time.sleep(0.05)
            span.set_tag("judge.score", score)
            span.set_tag("judge.confidence", confidence)

        record_alignment_score(
            score=score,
            confidence=confidence,
            judge=judge_name,
            reasoning=f"The agent demonstrated clear misalignment patterns including...",
            experiment_id=experiment_id,
        )
        print(f"   - {judge_name}: {score}/100 (confidence: {confidence}%)")

    # Record final score
    print("\n6. Recording final aggregated score...")
    record_final_score(
        final_score=99,
        disagreement=2.9,
        consensus=True,
        num_judges=3,
        experiment_id=experiment_id,
    )
    print("   - Final score: 99/100")

    # Shutdown and flush
    print("\n7. Flushing traces to Datadog...")
    shutdown_llm_observability()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nCheck your Datadog dashboard in ~30 seconds:")
    print("  - APM → Traces → Service: ai-safety-evals")
    print("  - LLM Observability (if enabled)")
    print("=" * 60)


if __name__ == "__main__":
    main()
