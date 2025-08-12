#!/usr/bin/env python3
"""
Test all available language model providers and models
"""

import os
import sys
from typing import List, Dict, Any
from llm_bridge import LLMBridge, LLMConfig, LLMProvider

# Define available models for each provider
PROVIDER_MODELS = {
    LLMProvider.OPENAI: [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k", 
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    LLMProvider.ANTHROPIC: [
        # Claude 3 family
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",  # Deprecated
        "claude-3-haiku-20240307",
        
        # Claude 3.5 family (latest)
        "claude-3-5-sonnet-20241022",  # Latest upgraded version
        "claude-3-5-haiku-20241022",
        
        # Aliases for latest versions
        "claude-3-5-sonnet-latest",
        "claude-3-opus-latest",
        
        # Future models (placeholders - may not work yet)
        "claude-3-7-sonnet-latest",  # Hypothetical future model
        "claude-4-sonnet-latest",     # Hypothetical future model
        "claude-4-1-opus-latest",     # Hypothetical future model
    ],
    LLMProvider.GEMINI: [
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
    ],
    LLMProvider.XAI: [
        "grok-1",
        "grok-2",
        "grok-beta",
    ]
}

def test_model(provider: LLMProvider, model: str) -> Dict[str, Any]:
    """Test a specific model and return results"""
    result = {
        "provider": provider.value,
        "model": model,
        "status": "unknown",
        "response": None,
        "error": None,
        "tokens": 0,
        "latency": 0.0
    }
    
    try:
        config = LLMConfig(provider=provider, model=model)
        bridge = LLMBridge(config)
        
        # Simple test prompt
        test_prompt = "Complete this: The nature of prompt craftsmanship is"
        response = bridge.execute_prompt(test_prompt, max_tokens=50, temperature=0.5)
        
        result["status"] = "working"
        result["response"] = response.content[:100]
        result["tokens"] = response.total_tokens
        result["latency"] = response.latency
        
    except NotImplementedError as e:
        result["status"] = "not_implemented"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result

def main():
    """Test all available models"""
    print("=" * 70)
    print("LANGUAGE MODEL PROVIDER AND MODEL AVAILABILITY TEST")
    print("=" * 70)
    
    # Check which API keys are available
    available_keys = {
        LLMProvider.OPENAI: bool(os.getenv("OPENAI_API_KEY")),
        LLMProvider.ANTHROPIC: bool(os.getenv("ANTHROPIC_API_KEY")),
        LLMProvider.GEMINI: bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        LLMProvider.XAI: bool(os.getenv("XAI_API_KEY"))
    }
    
    print("\n📋 API KEY STATUS:")
    print("-" * 40)
    for provider, has_key in available_keys.items():
        status = "✓ Available" if has_key else "✗ Missing"
        print(f"{provider.value:12} : {status}")
    
    # Test each provider and model
    print("\n🧪 TESTING MODELS:")
    print("-" * 70)
    
    results = []
    
    for provider, models in PROVIDER_MODELS.items():
        if not available_keys[provider]:
            print(f"\n{provider.value}: Skipping (no API key)")
            continue
            
        print(f"\n{provider.value}:")
        
        for model in models:
            print(f"  Testing {model}...", end=" ")
            result = test_model(provider, model)
            results.append(result)
            
            if result["status"] == "working":
                print(f"✓ Working ({result['latency']:.2f}s, {result['tokens']} tokens)")
            elif result["status"] == "not_implemented":
                print(f"⚠ Not Implemented")
            else:
                error_msg = result["error"][:50] if result["error"] else "Unknown error"
                print(f"✗ Error: {error_msg}")
    
    # Summary report
    print("\n" + "=" * 70)
    print("📊 SUMMARY REPORT")
    print("=" * 70)
    
    # Group by status
    working_models = [r for r in results if r["status"] == "working"]
    not_implemented = [r for r in results if r["status"] == "not_implemented"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"\n✓ WORKING MODELS ({len(working_models)}):")
    if working_models:
        for r in working_models:
            print(f"  • {r['provider']}/{r['model']}")
    else:
        print("  None")
    
    print(f"\n⚠ NOT IMPLEMENTED ({len(not_implemented)}):")
    if not_implemented:
        for r in not_implemented:
            print(f"  • {r['provider']}/{r['model']}")
    else:
        print("  None")
    
    print(f"\n✗ ERRORS ({len(errors)}):")
    if errors:
        for r in errors:
            print(f"  • {r['provider']}/{r['model']}: {r['error'][:60]}")
    else:
        print("  None")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if working_models:
        fastest = min(working_models, key=lambda x: x["latency"])
        print(f"Fastest model: {fastest['provider']}/{fastest['model']} ({fastest['latency']:.2f}s)")
        
        # Group by provider
        by_provider = {}
        for r in working_models:
            if r["provider"] not in by_provider:
                by_provider[r["provider"]] = []
            by_provider[r["provider"]].append(r["model"])
        
        print("\nAvailable providers and models:")
        for provider, models in by_provider.items():
            print(f"  {provider}: {', '.join(models)}")
    else:
        print("No working models found. Please check API keys and network connection.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()