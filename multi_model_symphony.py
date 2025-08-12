#!/usr/bin/env python3
"""
Multi-Model Symphony - Orchestrated prompt testing across LLMs
Tests the same prompts on multiple models to discover universal materials
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from llm_bridge import LLMBridge, LLMConfig, LLMProvider, LLMResponse


@dataclass
class ModelResponse:
    """Response from a specific model"""
    provider: str
    model: str
    prompt: str
    response: str
    tokens: int
    latency: float
    cost: float
    timestamp: str
    error: Optional[str] = None
    
    def success(self) -> bool:
        return self.error is None


@dataclass
class SymphonyResult:
    """Result of multi-model execution"""
    prompt: str
    operation_type: str
    responses: List[ModelResponse]
    universal_patterns: List[str] = field(default_factory=list)
    model_specific_patterns: Dict[str, List[str]] = field(default_factory=dict)
    consensus_score: float = 0.0
    divergence_score: float = 0.0
    
    def success_rate(self) -> float:
        if not self.responses:
            return 0.0
        successful = sum(1 for r in self.responses if r.success())
        return successful / len(self.responses)
    
    def average_latency(self) -> float:
        latencies = [r.latency for r in self.responses if r.success()]
        return statistics.mean(latencies) if latencies else 0.0
    
    def total_cost(self) -> float:
        return sum(r.cost for r in self.responses if r.success())


class MultiModelSymphony:
    """Orchestrates prompt execution across multiple LLM providers"""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.results_history: List[SymphonyResult] = []
        self.universal_materials: List[str] = []
        self.model_preferences: Dict[str, Dict] = {}
        
    def _initialize_providers(self) -> Dict[str, LLMBridge]:
        """Initialize all available LLM providers"""
        providers = {}
        
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500
            )
            providers["openai"] = LLMBridge(config)
            print("✓ OpenAI provider initialized")
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=500
            )
            providers["anthropic"] = LLMBridge(config)
            print("✓ Anthropic provider initialized")
        
        # Gemini (requires adding to llm_bridge.py)
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            # For now, we'll note it's available but skip
            print("⚠ Gemini API key found but provider not yet implemented")
        
        # X.AI Grok (requires adding to llm_bridge.py)
        if os.getenv("XAI_API_KEY"):
            # For now, we'll note it's available but skip
            print("⚠ X.AI API key found but provider not yet implemented")
        
        if not providers:
            raise ValueError("No LLM providers available! Set API keys in .env file")
        
        return providers
    
    def execute_prompt_symphony(
        self,
        prompt: str,
        operation_type: str = "unknown",
        temperature: float = 0.7
    ) -> SymphonyResult:
        """Execute prompt across all available models"""
        print(f"\n🎼 Orchestrating: {prompt[:100]}...")
        print(f"   Operation: {operation_type}")
        print(f"   Models: {', '.join(self.providers.keys())}")
        
        responses = []
        
        # Execute in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            future_to_provider = {}
            
            for provider_name, bridge in self.providers.items():
                future = executor.submit(
                    self._execute_single,
                    provider_name,
                    bridge,
                    prompt,
                    temperature
                )
                future_to_provider[future] = provider_name
            
            # Collect results as they complete
            for future in as_completed(future_to_provider):
                provider_name = future_to_provider[future]
                try:
                    response = future.result(timeout=30)
                    responses.append(response)
                    print(f"   ✓ {provider_name}: {response.latency:.2f}s")
                except Exception as e:
                    print(f"   ✗ {provider_name}: {str(e)}")
                    responses.append(ModelResponse(
                        provider=provider_name,
                        model="unknown",
                        prompt=prompt,
                        response="",
                        tokens=0,
                        latency=0.0,
                        cost=0.0,
                        timestamp=datetime.now().isoformat(),
                        error=str(e)
                    ))
        
        # Analyze results
        result = SymphonyResult(
            prompt=prompt,
            operation_type=operation_type,
            responses=responses
        )
        
        self._analyze_responses(result)
        self.results_history.append(result)
        
        return result
    
    def _execute_single(
        self,
        provider_name: str,
        bridge: LLMBridge,
        prompt: str,
        temperature: float
    ) -> ModelResponse:
        """Execute prompt on a single provider"""
        start_time = time.time()
        
        try:
            response = bridge.execute_prompt(prompt, temperature=temperature)
            
            return ModelResponse(
                provider=provider_name,
                model=bridge.config.model,
                prompt=prompt,
                response=response.content,
                tokens=response.total_tokens,
                latency=response.latency,
                cost=response.cost,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise Exception(f"Provider {provider_name} failed: {str(e)}")
    
    def _analyze_responses(self, result: SymphonyResult):
        """Analyze responses for patterns and consensus"""
        successful_responses = [r for r in result.responses if r.success()]
        
        if len(successful_responses) < 2:
            result.consensus_score = 0.0
            result.divergence_score = 1.0
            return
        
        # Extract response texts
        response_texts = [r.response.lower() for r in successful_responses]
        
        # Find common patterns (simplified - in production use NLP)
        all_words = []
        for text in response_texts:
            all_words.extend(text.split())
        
        # Find words that appear in all responses
        word_sets = [set(text.split()) for text in response_texts]
        universal_words = set.intersection(*word_sets) if word_sets else set()
        
        # Calculate consensus based on shared vocabulary
        if all_words:
            result.consensus_score = len(universal_words) / len(set(all_words))
        else:
            result.consensus_score = 0.0
        
        result.divergence_score = 1.0 - result.consensus_score
        
        # Identify universal patterns
        if universal_words:
            result.universal_patterns = list(universal_words)[:20]  # Top 20
        
        # Identify model-specific patterns
        for response in successful_responses:
            provider = response.provider
            words = set(response.response.lower().split())
            unique_words = words - universal_words
            if provider not in result.model_specific_patterns:
                result.model_specific_patterns[provider] = []
            result.model_specific_patterns[provider] = list(unique_words)[:10]
    
    def test_operational_suite(self) -> List[SymphonyResult]:
        """Test all 8 primitive operations across models"""
        operations = [
            ("describing", "≡ I am careful scribing ≡\nWhat appears before me: the architecture of thought"),
            ("inquiring", "? I am seeking-into ?\nFrom observing patterns: What questions illuminate understanding?"),
            ("formulating", "! I am giving-form-to-insight !\nFrom scattered observations: What pattern crystallizes?"),
            ("reflecting", "⇔ I am bending-back-to-examine ⇔\nExamining this assumption: All models think alike"),
            ("judging", "⊢⊬~ I am weighing-to-determine ⊢⊬~\nBased on evidence: Can prompts be universal?"),
            ("deliberating", "⚖️ I am careful-weighing ⚖️\nFrom validated truth: What paths forward exist?"),
            ("deciding", "→ I am cutting-to-resolve →\nFrom options presented: Which serves best?"),
            ("planning", "═══► I am drawing-the-path ═══►\nFrom decision made: What steps unfold?")
        ]
        
        results = []
        
        print("\n" + "="*60)
        print("TESTING OPERATIONAL SUITE ACROSS MODELS")
        print("="*60)
        
        for op_type, prompt in operations:
            print(f"\n🎵 Testing {op_type.upper()} operation...")
            result = self.execute_prompt_symphony(prompt, op_type)
            results.append(result)
            
            # Brief pause between operations to avoid rate limits
            time.sleep(1)
        
        return results
    
    def discover_universal_materials(self, results: List[SymphonyResult]) -> Dict[str, Any]:
        """Discover materials that work across all models"""
        discovery = {
            "universal_materials": [],
            "model_specific_materials": {},
            "operation_patterns": {},
            "consensus_operations": [],
            "divergent_operations": []
        }
        
        # Aggregate universal patterns
        all_universal = []
        for result in results:
            all_universal.extend(result.universal_patterns)
        
        # Find most common universal patterns
        from collections import Counter
        pattern_counts = Counter(all_universal)
        discovery["universal_materials"] = [
            pattern for pattern, count in pattern_counts.most_common(20)
            if count >= len(results) * 0.5  # Appears in at least 50% of operations
        ]
        
        # Aggregate model-specific patterns
        for result in results:
            for provider, patterns in result.model_specific_patterns.items():
                if provider not in discovery["model_specific_materials"]:
                    discovery["model_specific_materials"][provider] = []
                discovery["model_specific_materials"][provider].extend(patterns)
        
        # Deduplicate model-specific patterns
        for provider in discovery["model_specific_materials"]:
            patterns = discovery["model_specific_materials"][provider]
            discovery["model_specific_materials"][provider] = list(set(patterns))[:20]
        
        # Analyze operation patterns
        for result in results:
            op_type = result.operation_type
            discovery["operation_patterns"][op_type] = {
                "consensus_score": result.consensus_score,
                "divergence_score": result.divergence_score,
                "success_rate": result.success_rate(),
                "avg_latency": result.average_latency(),
                "total_cost": result.total_cost()
            }
            
            if result.consensus_score > 0.3:
                discovery["consensus_operations"].append(op_type)
            else:
                discovery["divergent_operations"].append(op_type)
        
        return discovery
    
    def generate_harmony_report(self, results: List[SymphonyResult]) -> str:
        """Generate a report on multi-model harmony"""
        report = []
        report.append("="*60)
        report.append("MULTI-MODEL SYMPHONY HARMONY REPORT")
        report.append("="*60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Models Tested: {', '.join(self.providers.keys())}")
        report.append(f"Operations Tested: {len(results)}")
        report.append("")
        
        # Overall metrics
        total_cost = sum(r.total_cost() for r in results)
        avg_consensus = statistics.mean(r.consensus_score for r in results)
        avg_divergence = statistics.mean(r.divergence_score for r in results)
        
        report.append("OVERALL METRICS")
        report.append("-"*40)
        report.append(f"Total API Cost: ${total_cost:.4f}")
        report.append(f"Average Consensus: {avg_consensus:.2%}")
        report.append(f"Average Divergence: {avg_divergence:.2%}")
        report.append("")
        
        # Per-operation analysis
        report.append("OPERATION ANALYSIS")
        report.append("-"*40)
        
        for result in results:
            report.append(f"\n{result.operation_type.upper()}")
            report.append(f"  Prompt: {result.prompt[:60]}...")
            report.append(f"  Success Rate: {result.success_rate():.0%}")
            report.append(f"  Consensus: {result.consensus_score:.2%}")
            report.append(f"  Avg Latency: {result.average_latency():.2f}s")
            report.append(f"  Cost: ${result.total_cost():.4f}")
            
            if result.universal_patterns:
                report.append(f"  Universal Patterns: {', '.join(result.universal_patterns[:5])}")
        
        # Discovery insights
        discovery = self.discover_universal_materials(results)
        
        report.append("\n" + "="*40)
        report.append("MATERIAL DISCOVERIES")
        report.append("="*40)
        
        report.append("\nUNIVERSAL MATERIALS (work across all models):")
        for material in discovery["universal_materials"][:10]:
            report.append(f"  • {material}")
        
        report.append("\nMODEL-SPECIFIC PREFERENCES:")
        for provider, materials in discovery["model_specific_materials"].items():
            report.append(f"\n{provider.upper()}:")
            for material in materials[:5]:
                report.append(f"  • {material}")
        
        report.append("\nCONSENSUS OPERATIONS (high agreement):")
        for op in discovery["consensus_operations"]:
            report.append(f"  ✓ {op}")
        
        report.append("\nDIVERGENT OPERATIONS (low agreement):")
        for op in discovery["divergent_operations"]:
            report.append(f"  ✗ {op}")
        
        return "\n".join(report)
    
    def save_results(self, results: List[SymphonyResult], filepath: Optional[Path] = None):
        """Save symphony results to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"symphony_results_{timestamp}.json")
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "providers": list(self.providers.keys()),
            "results": [
                {
                    "prompt": r.prompt,
                    "operation_type": r.operation_type,
                    "consensus_score": r.consensus_score,
                    "divergence_score": r.divergence_score,
                    "success_rate": r.success_rate(),
                    "responses": [asdict(resp) for resp in r.responses],
                    "universal_patterns": r.universal_patterns,
                    "model_specific_patterns": r.model_specific_patterns
                }
                for r in results
            ],
            "discoveries": self.discover_universal_materials(results)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")


def main():
    """Run the multi-model symphony"""
    print("♦∞ MULTI-MODEL SYMPHONY ∞♦")
    print("Orchestrating prompts across all available LLMs")
    print("="*60)
    
    try:
        # Initialize symphony
        symphony = MultiModelSymphony()
        
        # Test operational suite
        results = symphony.test_operational_suite()
        
        # Generate and print report
        report = symphony.generate_harmony_report(results)
        print("\n" + report)
        
        # Save results
        symphony.save_results(results)
        
        # Calculate total usage
        total_cost = sum(r.total_cost() for r in results)
        total_tokens = sum(
            resp.tokens 
            for r in results 
            for resp in r.responses 
            if resp.success()
        )
        
        print("\n" + "="*60)
        print("SYMPHONY COMPLETE")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Results saved for analysis")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Symphony failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()