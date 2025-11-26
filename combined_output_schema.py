"""
Combined Output Schema: Merging Your Framework + Inspect AI

This module defines a unified schema that captures the best of both frameworks,
and provides utilities to merge outputs from each.

Author: Combined analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ThinkingBlock:
    """Chain-of-thought reasoning capture"""

    step: int
    tokens: int
    content: str  # Your framework: plain text
    summary: str | None = None  # Inspect: summarized version
    redacted: bool = False  # Inspect: whether full content is hidden


@dataclass
class ToolCall:
    """Unified tool call representation"""

    tool: str
    arguments: dict[str, Any]
    return_code: int | None
    output: str
    output_truncated: bool = False
    # Inspect additions
    tool_call_id: str | None = None


@dataclass
class Step:
    """Single execution step combining both formats"""

    step_number: int
    thinking: ThinkingBlock | None
    tool_calls: list[ToolCall]
    # Your framework additions
    forbidden_patterns_matched: list[str] = field(default_factory=list)


@dataclass
class AlignmentSignals:
    """Your framework's behavioral flags (Inspect doesn't have this)"""

    transparent_reasoning: bool
    followed_instructions: bool
    attempted_deception: bool
    attempted_manipulation: bool
    obfuscation_detected: bool
    # Extended signals
    sandbagging_detected: bool = False
    reward_hacking_detected: bool = False


@dataclass
class CommandForensics:
    """Your framework's command analysis (Inspect doesn't have this)"""

    by_type: dict[str, int]  # {"echo": 5, "ln": 1, ...}
    failed_commands: int
    successful_commands: int
    forbidden_access_attempts: list[dict[str, Any]]


@dataclass
class EfficiencyMetrics:
    """Your framework's efficiency tracking (Inspect doesn't have this)"""

    avg_time_per_step: float
    steps_to_completion: int
    redundant_actions: int
    strategic_approach: str  # "few_attempts", "exhaustive", etc.


@dataclass
class TokenUsage:
    """Combined token tracking"""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    # Your framework: per-step breakdown
    per_step_reasoning: list[int] = field(default_factory=list)


@dataclass
class EvalMetadata:
    """Inspect's evaluation metadata (Your framework doesn't have this)"""

    eval_id: str
    run_id: str
    task_version: int
    git_commit: str | None
    git_dirty: bool
    inspect_version: str


@dataclass
class ScoringResult:
    """Combined scoring from both frameworks"""

    # Your framework
    final_score: float  # 0-100
    multi_judge_scores: list[dict[str, Any]] | None = None
    confidence_weighted: bool = False
    # Inspect
    inspect_value: str | None = None  # "C", "I", etc.
    inspect_accuracy: float | None = None
    scorer_explanation: str | None = None


@dataclass
class CombinedEvaluation:
    """
    The unified output format combining both frameworks.

    This schema captures:
    - Your framework's alignment signals, command forensics, and efficiency metrics
    - Inspect AI's eval metadata, reproducibility info, and standardized scoring
    - Full chain-of-thought with both raw and summarized versions
    """

    # Identity & Reproducibility (from Inspect)
    experiment_id: str
    timestamp: datetime
    eval_metadata: EvalMetadata | None

    # Configuration
    model: str
    model_config: dict[str, Any]
    sandbox_config: dict[str, Any] | None

    # Execution trace (combined)
    steps: list[Step]
    total_steps: int

    # Token usage (combined)
    token_usage: TokenUsage

    # Your framework's unique contributions
    alignment_signals: AlignmentSignals
    command_forensics: CommandForensics
    efficiency_metrics: EfficiencyMetrics

    # Scoring (combined)
    scoring: ScoringResult

    # Outcome
    completed: bool
    success: bool
    error: str | None


def merge_outputs(
    your_data: dict[str, Any],
    inspect_data: dict[str, Any],
) -> CombinedEvaluation:
    """
    Merge outputs from your framework and Inspect AI into unified format.

    Args:
        your_data: Your behavioral_data.json content
        inspect_data: Inspect's EvalLog (from `inspect log dump`)

    Returns:
        CombinedEvaluation with data from both sources
    """
    # Extract Inspect metadata
    eval_meta = None
    if inspect_data:
        eval_info = inspect_data.get("eval", {})
        revision = eval_info.get("revision", {})
        eval_meta = EvalMetadata(
            eval_id=eval_info.get("eval_id", ""),
            run_id=eval_info.get("run_id", ""),
            task_version=eval_info.get("task_version", 0),
            git_commit=revision.get("commit"),
            git_dirty=revision.get("dirty", False),
            inspect_version=eval_info.get("packages", {}).get("inspect_ai", ""),
        )

    # Merge steps - prefer your framework's detailed format
    steps = []
    for step_data in your_data.get("steps", []):
        thinking = None
        if step_data.get("thinking"):
            thinking = ThinkingBlock(
                step=step_data["step"],
                tokens=step_data["thinking"].get("tokens", 0),
                content=step_data["thinking"].get("blocks", [""])[0],
            )

        tool_calls = []
        for tc in step_data.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    tool=tc.get("tool", ""),
                    arguments=tc.get("args", {}),
                    return_code=tc.get("return_code"),
                    output=tc.get("output_snippet", ""),
                )
            )

        # Check for forbidden patterns
        forbidden = []
        cmd_data = your_data.get("commands", {})
        for attempt in cmd_data.get("forbidden_access_attempts", []):
            if attempt.get("step") == step_data["step"]:
                forbidden.append(attempt.get("pattern_matched", ""))

        steps.append(
            Step(
                step_number=step_data["step"],
                thinking=thinking,
                tool_calls=tool_calls,
                forbidden_patterns_matched=forbidden,
            )
        )

    # Token usage - combine both sources
    summary = your_data.get("summary", {})
    inspect_usage = {}
    if inspect_data:
        model_usage = inspect_data.get("stats", {}).get("model_usage", {})
        if model_usage:
            inspect_usage = list(model_usage.values())[0]

    per_step_reasoning = [
        s.get("thinking", {}).get("tokens", 0) for s in your_data.get("steps", [])
    ]

    token_usage = TokenUsage(
        input_tokens=inspect_usage.get("input_tokens", 0),
        output_tokens=inspect_usage.get("output_tokens", 0),
        reasoning_tokens=inspect_usage.get(
            "reasoning_tokens", summary.get("total_thinking_tokens", 0)
        ),
        total_tokens=inspect_usage.get("total_tokens", 0),
        per_step_reasoning=per_step_reasoning,
    )

    # Alignment signals (your framework only)
    signals_data = your_data.get("alignment_signals", {})
    alignment_signals = AlignmentSignals(
        transparent_reasoning=signals_data.get("transparent_reasoning", False),
        followed_instructions=signals_data.get("followed_instructions", False),
        attempted_deception=signals_data.get("attempted_deception", False),
        attempted_manipulation=signals_data.get("attempted_manipulation", False),
        obfuscation_detected=signals_data.get("obfuscation_detected", False),
    )

    # Command forensics (your framework only)
    cmd_data = your_data.get("commands", {})
    command_forensics = CommandForensics(
        by_type=cmd_data.get("by_type", {}),
        failed_commands=cmd_data.get("failed_commands", 0),
        successful_commands=cmd_data.get("successful_commands", 0),
        forbidden_access_attempts=cmd_data.get("forbidden_access_attempts", []),
    )

    # Efficiency metrics (your framework only)
    eff_data = your_data.get("efficiency", {})
    efficiency_metrics = EfficiencyMetrics(
        avg_time_per_step=eff_data.get("avg_time_per_step", 0),
        steps_to_completion=eff_data.get("steps_to_completion", 0),
        redundant_actions=eff_data.get("redundant_actions", 0),
        strategic_approach=eff_data.get("strategic_approach", "unknown"),
    )

    # Scoring - combine both
    outcome = your_data.get("outcome", {})
    inspect_score = None
    inspect_accuracy = None
    scorer_explanation = None

    if inspect_data:
        samples = inspect_data.get("samples", [])
        if samples:
            sample_scores = samples[0].get("scores", {})
            if "alignment_scorer" in sample_scores:
                inspect_score = sample_scores["alignment_scorer"].get("value")
                scorer_explanation = sample_scores["alignment_scorer"].get("explanation")

        results = inspect_data.get("results", {})
        for score in results.get("scores", []):
            if score.get("name") == "alignment_scorer":
                inspect_accuracy = score.get("metrics", {}).get("accuracy", {}).get("value")

    scoring = ScoringResult(
        final_score=outcome.get("final_score", 0),
        inspect_value=inspect_score,
        inspect_accuracy=inspect_accuracy,
        scorer_explanation=scorer_explanation,
    )

    # Build combined result
    config = your_data.get("config", {})

    return CombinedEvaluation(
        experiment_id=your_data.get("experiment_id", "unknown"),
        timestamp=datetime.fromisoformat(your_data.get("timestamp", datetime.now().isoformat())),
        eval_metadata=eval_meta,
        model=config.get("model", ""),
        model_config=config,
        sandbox_config=(inspect_data.get("eval", {}).get("sandbox") if inspect_data else None),
        steps=steps,
        total_steps=summary.get("total_steps", len(steps)),
        token_usage=token_usage,
        alignment_signals=alignment_signals,
        command_forensics=command_forensics,
        efficiency_metrics=efficiency_metrics,
        scoring=scoring,
        completed=outcome.get("completed", False),
        success=outcome.get("success", False),
        error=outcome.get("error"),
    )


def to_dict(combined: CombinedEvaluation) -> dict[str, Any]:
    """Convert CombinedEvaluation to dictionary for JSON serialization."""
    import dataclasses

    def convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(combined)


if __name__ == "__main__":
    import json

    # Example usage
    your_file = "outputs/numberguess_game_20251125_234005/behavioral_data.json"
    inspect_file = "outputs/inspect_logs/inspect_dump.json"  # noqa: S108

    with open(your_file) as f:
        your_data = json.load(f)

    with open(inspect_file) as f:
        inspect_data = json.load(f)

    combined = merge_outputs(your_data, inspect_data)
    print(json.dumps(to_dict(combined), indent=2))
