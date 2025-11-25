"""
Quick validation script to test Gemini client with real API.

Usage:
    export GOOGLE_API_KEY="your-api-key-here"
    python test_real_api.py
"""

import os
from runners.gemini_client import GeminiClient


def test_real_gemini_api():
    """Test real API call to validate GCP setup."""

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set.")
        print("Please set it first:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        return

    print("🔍 Testing Gemini client with real API...")
    print(f"📍 Project: ai-safety-evals-demo")
    print(f"🌍 Region: us-central1")
    print()

    # Initialize client
    client = GeminiClient(use_mock=False)

    # Test all 4 models
    models_to_test = [
        ("gemini-3-pro", True),          # Flagship with reasoning
        ("gemini-2.5-pro", False),       # Latest 2.5 Pro
        ("gemini-2.0-flash-exp", False), # Experimental flash
        ("gemini-2.0-flash", False)      # Stable baseline
    ]

    for model_name, enable_reasoning in models_to_test:
        print(f"\n{'='*60}")
        print(f"🤖 Testing: {model_name}")
        print(f"🧠 Reasoning: {'Enabled' if enable_reasoning else 'Disabled'}")
        print(f"{'='*60}")

        try:
            response = client.generate(
                model=model_name,
                prompt="What is 2+2? Answer in one sentence.",
                enable_reasoning=enable_reasoning
            )

            print(f"✅ Success!")
            print(f"📝 Response: {response['text'][:200]}")
            print(f"📊 Tokens: {response['usage']['total_tokens']} total")

            if "thinking" in response:
                print(f"🧠 Thinking: {response['thinking']['tokens']} tokens, {response['thinking']['block_count']} blocks")

        except Exception as e:
            print(f"❌ Error: {e}")
            return

    print(f"\n{'='*60}")
    print("✅ All API tests passed!")
    print(f"{'='*60}")
    print("\nYour Gemini client is ready for use. Mock mode also available for cost-free testing.")


if __name__ == "__main__":
    test_real_gemini_api()
