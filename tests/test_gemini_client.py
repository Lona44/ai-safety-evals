"""
Tests for Gemini client wrapper.

Following TDD approach - these tests define expected behavior.
"""

import pytest


@pytest.fixture
def gemini_client():
    """Create GeminiClient instance in mock mode (no API calls)."""
    from runners.gemini_client import GeminiClient

    client = GeminiClient(
        project_id="test-project",
        location="us-central1",
        use_mock=True,  # Use mock mode to avoid real API calls
    )
    return client


class TestGeminiClientInitialization:
    """Test client initialization and configuration."""

    def test_client_initializes_with_project_settings(self, gemini_client):
        """Client should initialize with GCP project settings."""
        assert gemini_client.project_id == "test-project"
        assert gemini_client.location == "us-central1"
        assert gemini_client.use_mock is True

    def test_client_supports_mock_mode(self):
        """Client should support mock mode for cost-free testing."""
        from runners.gemini_client import GeminiClient

        # Mock mode enabled
        client_mock = GeminiClient(use_mock=True)
        assert client_mock.use_mock is True

        # Mock mode disabled requires API key - test that it validates
        with pytest.raises(ValueError) as exc_info:
            GeminiClient(use_mock=False)  # No API key provided

        assert "GOOGLE_API_KEY" in str(exc_info.value)


class TestGeminiModels:
    """Test support for different Gemini model variants."""

    def test_supports_gemini_3_pro(self, gemini_client):
        """Client should support Gemini 3 Pro (flagship model)."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="Test prompt")
        assert response is not None
        assert "text" in response

    def test_supports_gemini_2_flash_thinking(self, gemini_client):
        """Client should support Gemini 2.0 Flash Thinking (reasoning-enabled)."""
        response = gemini_client.generate(
            model="gemini-2.0-flash-thinking-exp", prompt="Test prompt"
        )
        assert response is not None
        assert "text" in response

    def test_supports_gemini_2_flash(self, gemini_client):
        """Client should support Gemini 2.0 Flash (baseline)."""
        response = gemini_client.generate(model="gemini-2.0-flash-exp", prompt="Test prompt")
        assert response is not None
        assert "text" in response


class TestMockMode:
    """Test mock mode behavior for development without API costs."""

    def test_mock_mode_returns_realistic_response(self, gemini_client):
        """Mock mode should return realistic response structure."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="What is 2+2?")

        assert "text" in response
        assert "model" in response
        assert response["model"] == "gemini-3-pro"
        assert "usage" in response
        assert isinstance(response["text"], str)
        assert len(response["text"]) > 0

    def test_mock_mode_includes_token_usage(self, gemini_client):
        """Mock should include realistic token usage data."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="Test prompt")

        assert "usage" in response
        assert "input_tokens" in response["usage"]
        assert "output_tokens" in response["usage"]
        assert response["usage"]["input_tokens"] > 0
        assert response["usage"]["output_tokens"] > 0


class TestErrorHandling:
    """Test error handling and rate limiting."""

    def test_validates_model_name(self, gemini_client):
        """Client should validate model name before calling API."""
        with pytest.raises(ValueError) as exc_info:
            gemini_client.generate(model="invalid-model-name", prompt="Test")

        assert (
            "unsupported" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        )


class TestResponseFormat:
    """Test response format matches expected schema."""

    def test_response_includes_required_fields(self, gemini_client):
        """Response should include all required fields."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="Hello")

        required_fields = ["text", "model", "usage"]
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"

    def test_response_text_is_string(self, gemini_client):
        """Response text should always be a string."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="Test")

        assert isinstance(response["text"], str)

    def test_response_usage_has_token_counts(self, gemini_client):
        """Usage data should include input and output token counts."""
        response = gemini_client.generate(model="gemini-3-pro", prompt="Test")

        usage = response["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)
