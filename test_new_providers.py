#!/usr/bin/env python3
"""
Test script for Gemini and X.AI providers
"""

from llm_bridge import LLMBridge, LLMConfig, LLMProvider


def test_provider(provider_name: str, model: str, test_prompt: str):
    """Test a specific provider and model"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} - {model}")
    print(f"{'='*60}")
    
    try:
        # Initialize the provider
        config = LLMConfig(
            provider=LLMProvider[provider_name.upper()],
            model=model,
            max_tokens=100,
            temperature=0.7
        )
        bridge = LLMBridge(config)
        
        # Execute test prompt
        print(f"Prompt: {test_prompt}")
        print("-" * 40)
        
        response = bridge.execute_prompt(test_prompt)
        
        print(f"Response: {response.content}")
        print("-" * 40)
        print(f"Model: {response.model}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Latency: {response.latency:.2f}s")
        print(f"Cost: ${response.cost:.4f}")
        
        if response.metadata:
            print(f"Metadata: {response.metadata}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main test function"""
    print("TESTING NEW PROVIDER IMPLEMENTATIONS")
    print("=" * 60)
    
    test_prompt = "What is the key to effective prompt engineering?"
    
    # Test Gemini models
    gemini_models = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp"
    ]
    
    print("\n📌 GEMINI PROVIDER TESTS")
    for model in gemini_models:
        test_provider("GEMINI", model, test_prompt)
    
    # Test X.AI models
    xai_models = [
        "grok-2",  # Only grok-2 works based on previous test
    ]
    
    print("\n📌 X.AI PROVIDER TESTS")
    for model in xai_models:
        test_provider("XAI", model, test_prompt)
    
    # Test switching providers
    print("\n📌 PROVIDER SWITCHING TEST")
    print("=" * 60)
    
    try:
        # Start with Gemini
        config = LLMConfig(provider=LLMProvider.GEMINI, model="gemini-1.5-flash")
        bridge = LLMBridge(config)
        
        print("Initial provider: GEMINI")
        response = bridge.execute_prompt("Hello, I am")
        print(f"Response: {response.content[:50]}...")
        
        # Switch to X.AI
        bridge.switch_provider(LLMProvider.XAI)
        bridge.set_model("grok-2")
        
        print("\nSwitched to: X.AI")
        response = bridge.execute_prompt("Hello, I am")
        print(f"Response: {response.content[:50]}...")
        
        # Get usage report
        print("\n📊 USAGE REPORT")
        print("-" * 40)
        report = bridge.get_usage_report()
        for key, value in report.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error in provider switching: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")


if __name__ == "__main__":
    main()