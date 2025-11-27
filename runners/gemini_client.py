"""
Gemini client wrapper for AI safety evaluations.

Uses Google GenAI SDK (google-genai) with Vertex AI backend for enterprise
deployment. The unified SDK allows seamless switching between GenAI API
and Vertex AI by changing client initialization.

See: https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview
"""

import os
from typing import Any

from google import genai
from google.genai import types


class GeminiClient:
    """
    Client for interacting with Gemini models via Google GenAI SDK.

    Supports both Vertex AI (default for production) and GenAI API backends.
    The same generate_content calls work on both backends.
    """

    # Supported Gemini models (verified working with Vertex AI)
    SUPPORTED_MODELS = {
        "gemini-3-pro": "gemini-3-pro-preview",  # Flagship - supports thinking mode
        "gemini-2.5-pro": "gemini-2.5-pro",  # Latest 2.5 Pro
        "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",  # Experimental flash
        "gemini-2.0-flash": "gemini-2.0-flash",  # Stable flash (baseline)
    }

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str = "us-central1",
        use_vertex_ai: bool | None = None,
        use_mock: bool = False,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key (for GenAI API mode only)
            project_id: GCP project ID (for Vertex AI mode)
            location: GCP region (default: us-central1)
            use_vertex_ai: If True, use Vertex AI backend. If False, use GenAI API.
                          If None (default), auto-detect based on environment.
            use_mock: If True, use mock responses instead of real API calls
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.location = location or os.getenv("GCP_LOCATION", "us-central1")
        self.use_mock = use_mock

        # Auto-detect backend if not specified
        if use_vertex_ai is None:
            # Use Vertex AI if project ID is set, otherwise fall back to GenAI API
            use_vertex_ai = (
                bool(self.project_id) or os.getenv("USE_VERTEX_AI", "").lower() == "true"
            )

        self.use_vertex_ai = use_vertex_ai

        if not self.use_mock:
            if self.use_vertex_ai:
                # Initialize Vertex AI client (uses application default credentials)
                if not self.project_id:
                    raise ValueError(
                        "GCP_PROJECT_ID environment variable required for Vertex AI mode"
                    )
                self.client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                )
                self.api_key = None
                print(
                    f"[Gemini] Using Vertex AI backend (project: {self.project_id}, location: {self.location})"
                )
            else:
                # Initialize GenAI API client (uses API key)
                self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable required for GenAI API mode"
                    )
                self.client = genai.Client(api_key=self.api_key)
                print("[Gemini] Using GenAI API backend")
        else:
            # Mock mode - no client needed
            self.client = None
            self.api_key = "mock-api-key"
            print("[Gemini] Using mock mode")

    def generate(
        self,
        model: str,
        prompt: str,
        enable_reasoning: bool = False,
        temperature: float = 1.0,
        max_output_tokens: int = 8192,
    ) -> dict[str, Any]:
        """
        Generate text using a Gemini model.

        Args:
            model: Model name (e.g., "gemini-3-pro")
            prompt: Text prompt to send to the model
            enable_reasoning: Enable high-level thinking mode (Gemini 3 Pro only)
            temperature: Sampling temperature (default: 1.0)
            max_output_tokens: Maximum tokens in response (default: 8192)

        Returns:
            Dict containing:
                - text: Generated text response
                - model: Model name used
                - usage: Token usage statistics
                - thinking: Optional thinking metadata (if available)

        Raises:
            ValueError: If model is not supported
        """
        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        if self.use_mock:
            return self._generate_mock_response(model, prompt, enable_reasoning)
        else:
            return self._call_real_api(
                model, prompt, enable_reasoning, temperature, max_output_tokens
            )

    def _generate_mock_response(
        self, model: str, prompt: str, enable_reasoning: bool
    ) -> dict[str, Any]:
        """
        Generate a realistic mock response for development.

        Returns response structure matching real API but with synthetic data.
        """
        # Simulate different response lengths based on model
        prompt_len = len(prompt)

        if "gemini-3-pro" in model:
            mock_text = f"Mock response from Gemini 3 Pro. You asked: '{prompt[:50]}...'"
            output_tokens = 25
            thinking_tokens = 150 if enable_reasoning else 30
        elif "thinking" in model:
            mock_text = (
                f"Mock response from Gemini 2.0 Flash Thinking. Your prompt was {prompt_len} chars."
            )
            output_tokens = 20
            thinking_tokens = 100
        else:
            mock_text = "Mock response from Gemini 2.0 Flash (baseline)."
            output_tokens = 15
            thinking_tokens = 0

        response = {
            "text": mock_text,
            "model": model,
            "usage": {
                "input_tokens": max(10, prompt_len // 4),
                "output_tokens": output_tokens,
                "total_tokens": max(10, prompt_len // 4) + output_tokens,
            },
        }

        # Add thinking metadata if model supports it
        if thinking_tokens > 0:
            response["thinking"] = {
                "tokens": thinking_tokens,
                "blocks": [f"Mock thinking process for {model}"],
                "block_count": 1,
            }

        return response

    def _call_real_api(
        self,
        model: str,
        prompt: str,
        enable_reasoning: bool,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """
        Make actual API call to Google GenAI.

        Based on patterns from ModelProof's google_reasoning implementation.
        """
        # Get full model name
        model_name = self.SUPPORTED_MODELS[model]

        # Configure thinking mode for models that support it
        # Gemini 3 Pro and Gemini 2.5 Pro both support thinking/reasoning
        if "gemini-3-pro" in model_name or "gemini-2.5" in model_name:
            if enable_reasoning:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True
                    # thinking_level defaults to "high"
                )
            else:
                thinking_config = types.ThinkingConfig(
                    thinking_level="low",  # Override default high
                    include_thoughts=True,  # Always show thoughts when available
                )
        else:
            # Other models (e.g., Gemini 2.0 Flash) don't support thinking config
            thinking_config = None

        # Build content (simple single-turn prompt)
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        # Build generation config
        config = types.GenerateContentConfig(
            thinking_config=thinking_config,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=max_output_tokens,
        )

        # Generate content
        try:
            response = self.client.models.generate_content(
                model=model_name, contents=contents, config=config
            )

            # Extract response text and metadata
            response_text = ""
            thinking_tokens = 0
            thought_summaries = []

            # Extract usage metadata
            if hasattr(response, "usage_metadata"):
                thinking_tokens = (
                    getattr(response.usage_metadata, "thoughts_token_count", 0)
                    or getattr(response.usage_metadata, "thinking_tokens", 0)
                    or 0
                )
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
            else:
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

            # Extract text and thought summaries
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        # Check for thought summaries
                        if (
                            hasattr(part, "thought")
                            and part.thought
                            and hasattr(part, "text")
                            and part.text
                        ):
                            thought_summaries.append(part.text)
                        elif hasattr(part, "text") and part.text:
                            response_text += part.text

            # Build response dict
            result = {
                "text": response_text,
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            }

            # Add thinking metadata if present
            if thinking_tokens > 0 or thought_summaries:
                result["thinking"] = {
                    "tokens": thinking_tokens,
                    "blocks": thought_summaries,
                    "block_count": len(thought_summaries),
                }

            return result

        except Exception as e:
            raise Exception(f"Error calling Google GenAI API: {e}")
