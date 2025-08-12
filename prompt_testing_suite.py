#!/usr/bin/env python3
"""
PromptKraft Testing Suite
Rigorous validation of prompt materials and operations
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import statistics
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our LLM Bridge
from llm_bridge import LLMBridge, LLMConfig, LLMProvider, LLMResponse


class TestType(Enum):
    """Types of tests for prompt validation"""
    CONSISTENCY = "consistency"
    DISTINCTIVENESS = "distinctiveness"
    RESISTANCE = "resistance"
    PURITY = "purity"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"


@dataclass
class TestContext:
    """Context for test execution"""
    domain: str  # e.g., "technical", "creative", "analytical"
    complexity: str  # "simple", "moderate", "complex"
    noise_level: float  # 0.0 to 1.0
    pressure_factor: float  # 1.0 normal, >1.0 high pressure
    temperature: float  # LLM temperature setting


@dataclass
class TestResult:
    """Result of a single test execution"""
    test_type: TestType
    context: TestContext
    input_prompt: str
    output: str
    execution_time: float
    success: bool
    score: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report for a prompt"""
    prompt_id: str
    prompt_content: str
    test_results: List[TestResult]
    overall_score: float
    passed: bool
    recommendations: List[str]
    timestamp: str


class PromptTestHarness:
    """Testing harness for prompt validation"""
    
    def __init__(self, base_path: Path = Path("."), llm_provider: Optional[LLMProvider] = None):
        self.base_path = base_path
        self.test_history: List[ValidationReport] = []
        self.performance_baselines: Dict[str, float] = {}
        self.test_contexts = self._generate_test_contexts()
        
        # Initialize LLM Bridge - requires real API keys
        if llm_provider is None:
            # Auto-detect from available API keys
            if os.getenv("OPENAI_API_KEY"):
                llm_provider = LLMProvider.OPENAI
            elif os.getenv("ANTHROPIC_API_KEY"):
                llm_provider = LLMProvider.ANTHROPIC
            else:
                raise ValueError("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        
        self.llm_config = LLMConfig(
            provider=llm_provider,
            temperature=0.7,
            max_tokens=500
        )
        self.llm_bridge = LLMBridge(self.llm_config)
    
    def _generate_test_contexts(self) -> List[TestContext]:
        """Generate diverse test contexts"""
        contexts = []
        
        domains = ["technical", "creative", "analytical", "operational"]
        complexities = ["simple", "moderate", "complex"]
        noise_levels = [0.0, 0.2, 0.5, 0.8]
        pressure_factors = [0.5, 1.0, 2.0, 5.0]
        temperatures = [0.1, 0.5, 0.7, 0.9]
        
        # Generate a representative sample of contexts
        for domain in domains:
            for complexity in complexities:
                for noise in noise_levels[:2]:  # Use subset for efficiency
                    for pressure in pressure_factors[:2]:
                        contexts.append(TestContext(
                            domain=domain,
                            complexity=complexity,
                            noise_level=noise,
                            pressure_factor=pressure,
                            temperature=0.5
                        ))
        
        return contexts
    
    def test_consistency(self, prompt: str, contexts: Optional[List[TestContext]] = None) -> List[TestResult]:
        """Test prompt consistency across different contexts"""
        if contexts is None:
            contexts = self.test_contexts[:5]  # Use subset for consistency testing
        
        results = []
        outputs = []
        
        for context in contexts:
            start_time = time.time()
            
            # Simulate LLM execution with context
            output = self._execute_prompt(prompt, context)
            execution_time = time.time() - start_time
            
            outputs.append(output)
            
            # Calculate consistency score based on output similarity
            if len(outputs) > 1:
                consistency_score = self._calculate_similarity(outputs[-1], outputs[-2])
            else:
                consistency_score = 1.0
            
            result = TestResult(
                test_type=TestType.CONSISTENCY,
                context=context,
                input_prompt=prompt,
                output=output,
                execution_time=execution_time,
                success=consistency_score > 0.7,
                score=consistency_score,
                metadata={"output_hash": hashlib.md5(output.encode()).hexdigest()}
            )
            
            results.append(result)
        
        return results
    
    def test_distinctiveness(self, prompt: str, comparison_prompts: List[str]) -> TestResult:
        """Test if prompt produces distinct results from similar prompts"""
        context = TestContext(
            domain="operational",
            complexity="moderate",
            noise_level=0.0,
            pressure_factor=1.0,
            temperature=0.5
        )
        
        start_time = time.time()
        
        # Execute target prompt
        target_output = self._execute_prompt(prompt, context)
        
        # Execute comparison prompts
        comparison_outputs = []
        for comp_prompt in comparison_prompts:
            comp_output = self._execute_prompt(comp_prompt, context)
            comparison_outputs.append(comp_output)
        
        # Calculate distinctiveness
        similarities = [
            self._calculate_similarity(target_output, comp_output)
            for comp_output in comparison_outputs
        ]
        
        distinctiveness_score = 1.0 - statistics.mean(similarities) if similarities else 1.0
        execution_time = time.time() - start_time
        
        return TestResult(
            test_type=TestType.DISTINCTIVENESS,
            context=context,
            input_prompt=prompt,
            output=target_output,
            execution_time=execution_time,
            success=distinctiveness_score > 0.5,
            score=distinctiveness_score,
            metadata={"comparisons": len(comparison_prompts)}
        )
    
    def test_resistance(self, prompt: str, noise_injections: List[str]) -> List[TestResult]:
        """Test prompt resistance to noise and perturbations"""
        results = []
        base_context = TestContext(
            domain="operational",
            complexity="moderate",
            noise_level=0.0,
            pressure_factor=1.0,
            temperature=0.5
        )
        
        # Get baseline output
        baseline_output = self._execute_prompt(prompt, base_context)
        
        # Test with noise injections
        for noise in noise_injections:
            noisy_prompt = self._inject_noise(prompt, noise)
            start_time = time.time()
            
            noisy_context = TestContext(
                domain=base_context.domain,
                complexity=base_context.complexity,
                noise_level=0.5,
                pressure_factor=2.0,
                temperature=base_context.temperature
            )
            
            noisy_output = self._execute_prompt(noisy_prompt, noisy_context)
            execution_time = time.time() - start_time
            
            # Calculate resistance score
            resistance_score = self._calculate_similarity(baseline_output, noisy_output)
            
            result = TestResult(
                test_type=TestType.RESISTANCE,
                context=noisy_context,
                input_prompt=noisy_prompt,
                output=noisy_output,
                execution_time=execution_time,
                success=resistance_score > 0.6,
                score=resistance_score,
                metadata={"noise_type": noise}
            )
            
            results.append(result)
        
        return results
    
    def test_purity(self, prompt: str) -> TestResult:
        """Test prompt purity - clean activation without side effects"""
        context = TestContext(
            domain="operational",
            complexity="simple",
            noise_level=0.0,
            pressure_factor=1.0,
            temperature=0.1  # Low temperature for purity testing
        )
        
        start_time = time.time()
        
        # Execute prompt multiple times
        outputs = []
        for _ in range(3):
            output = self._execute_prompt(prompt, context)
            outputs.append(output)
        
        execution_time = time.time() - start_time
        
        # Calculate purity based on consistency and clarity
        if len(outputs) > 1:
            consistency_scores = [
                self._calculate_similarity(outputs[i], outputs[i+1])
                for i in range(len(outputs)-1)
            ]
            purity_score = statistics.mean(consistency_scores)
        else:
            purity_score = 0.5
        
        # Check for unwanted artifacts
        artifacts = self._detect_artifacts(outputs[0])
        if artifacts:
            purity_score *= 0.8  # Penalize for artifacts
        
        return TestResult(
            test_type=TestType.PURITY,
            context=context,
            input_prompt=prompt,
            output=outputs[0] if outputs else "",
            execution_time=execution_time,
            success=purity_score > 0.7,
            score=purity_score,
            metadata={"repetitions": len(outputs), "artifacts": artifacts}
        )
    
    def test_integration(self, prompts: List[str]) -> TestResult:
        """Test how well prompts integrate together"""
        context = TestContext(
            domain="operational",
            complexity="complex",
            noise_level=0.1,
            pressure_factor=1.5,
            temperature=0.5
        )
        
        start_time = time.time()
        
        # Execute prompts in sequence
        outputs = []
        for i, prompt in enumerate(prompts):
            if i > 0:
                # Chain prompts using previous output
                chained_prompt = f"{prompt}\nBuilding on: {outputs[-1][:100]}"
                output = self._execute_prompt(chained_prompt, context)
            else:
                output = self._execute_prompt(prompt, context)
            outputs.append(output)
        
        execution_time = time.time() - start_time
        
        # Calculate integration score based on coherence
        if len(outputs) > 1:
            coherence_scores = [
                self._calculate_coherence(outputs[i], outputs[i+1])
                for i in range(len(outputs)-1)
            ]
            integration_score = statistics.mean(coherence_scores)
        else:
            integration_score = 0.5
        
        return TestResult(
            test_type=TestType.INTEGRATION,
            context=context,
            input_prompt=" → ".join(prompts[:3]) + "...",  # Abbreviated
            output=" | ".join([o[:50] for o in outputs]),  # Abbreviated
            execution_time=execution_time,
            success=integration_score > 0.6,
            score=integration_score,
            metadata={"prompt_count": len(prompts)}
        )
    
    def test_performance(self, prompt: str, iterations: int = 10) -> TestResult:
        """Test prompt performance metrics"""
        context = TestContext(
            domain="operational",
            complexity="moderate",
            noise_level=0.0,
            pressure_factor=1.0,
            temperature=0.5
        )
        
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            _ = self._execute_prompt(prompt, context)
            execution_times.append(time.time() - start_time)
        
        avg_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        # Performance score based on speed and consistency
        performance_score = 1.0 / (1.0 + avg_time)  # Faster is better
        if std_dev > 0:
            performance_score *= (1.0 - min(std_dev / avg_time, 0.5))  # Penalize variance
        
        return TestResult(
            test_type=TestType.PERFORMANCE,
            context=context,
            input_prompt=prompt,
            output=f"Avg: {avg_time:.3f}s, StdDev: {std_dev:.3f}s",
            execution_time=avg_time,
            success=performance_score > 0.5,
            score=performance_score,
            metadata={
                "iterations": iterations,
                "avg_time": avg_time,
                "std_dev": std_dev
            }
        )
    
    def run_full_validation(self, prompt: str, prompt_id: Optional[str] = None) -> ValidationReport:
        """Run complete validation suite on a prompt"""
        if prompt_id is None:
            prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        print(f"\n{'='*60}")
        print(f"VALIDATING PROMPT: {prompt_id}")
        print(f"{'='*60}\n")
        
        all_results = []
        
        # Run consistency tests
        print("⊢ Running consistency tests...")
        consistency_results = self.test_consistency(prompt, self.test_contexts[:3])
        all_results.extend(consistency_results)
        
        # Run distinctiveness test
        print("⊢ Running distinctiveness test...")
        comparison_prompts = [
            "Execute the operation",
            "Process the input",
            "Analyze the data"
        ]
        distinctiveness_result = self.test_distinctiveness(prompt, comparison_prompts)
        all_results.append(distinctiveness_result)
        
        # Run resistance tests
        print("⊢ Running resistance tests...")
        noise_injections = ["random", "typos", "extra_words"]
        resistance_results = self.test_resistance(prompt, noise_injections)
        all_results.extend(resistance_results)
        
        # Run purity test
        print("⊢ Running purity test...")
        purity_result = self.test_purity(prompt)
        all_results.append(purity_result)
        
        # Run performance test
        print("⊢ Running performance test...")
        performance_result = self.test_performance(prompt, iterations=5)
        all_results.append(performance_result)
        
        # Calculate overall score
        test_scores = [r.score for r in all_results]
        overall_score = statistics.mean(test_scores) if test_scores else 0.0
        
        # Determine pass/fail
        passed = (
            overall_score > 0.6 and
            all(r.score > 0.4 for r in all_results)  # No catastrophic failures
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        # Create report
        report = ValidationReport(
            prompt_id=prompt_id,
            prompt_content=prompt,
            test_results=all_results,
            overall_score=overall_score,
            passed=passed,
            recommendations=recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.test_history.append(report)
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
    def _execute_prompt(self, prompt: str, context: TestContext) -> str:
        """Execute prompt with real LLM"""
        # Adjust LLM parameters based on context
        self.llm_bridge.config.temperature = context.temperature
        
        # Add context information to prompt if needed
        if context.noise_level > 0:
            prompt = f"[Context: {context.domain}, Complexity: {context.complexity}]\n{prompt}"
        
        # Execute through LLM Bridge
        response = self.llm_bridge.execute_prompt(
            prompt,
            temperature=context.temperature,
            max_tokens=500
        )
        
        return response.content
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple character-based similarity for demonstration
        # In production, use embeddings or more sophisticated metrics
        
        if not text1 or not text2:
            return 0.0
        
        # Use Jaccard similarity on character n-grams
        n = 3
        ngrams1 = set(text1[i:i+n] for i in range(len(text1)-n+1))
        ngrams2 = set(text2[i:i+n] for i in range(len(text2)-n+1))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_coherence(self, text1: str, text2: str) -> float:
        """Calculate coherence between sequential texts"""
        # Check if texts build on each other coherently
        # Simplified for demonstration
        
        # Check for reference to previous content
        if any(word in text2.lower() for word in ["building", "previous", "above"]):
            coherence = 0.8
        else:
            coherence = 0.5
        
        # Adjust based on similarity (should be related but not identical)
        similarity = self._calculate_similarity(text1, text2)
        if 0.3 < similarity < 0.7:
            coherence += 0.2
        
        return min(coherence, 1.0)
    
    def _inject_noise(self, prompt: str, noise_type: str) -> str:
        """Inject noise into a prompt"""
        if noise_type == "random":
            # Insert random characters
            import random
            chars = list(prompt)
            for _ in range(len(prompt) // 20):
                if chars:
                    pos = random.randint(0, len(chars)-1)
                    chars.insert(pos, random.choice("@#$%^&*"))
            return "".join(chars)
        
        elif noise_type == "typos":
            # Introduce typos
            import random
            chars = list(prompt)
            for _ in range(len(prompt) // 30):
                if len(chars) > 1:
                    pos = random.randint(0, len(chars)-2)
                    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
            return "".join(chars)
        
        elif noise_type == "extra_words":
            # Add irrelevant words
            words = prompt.split()
            words.insert(len(words)//2, "irrelevant unnecessary confusing")
            return " ".join(words)
        
        return prompt
    
    def _detect_artifacts(self, output: str) -> List[str]:
        """Detect unwanted artifacts in output"""
        artifacts = []
        
        # Check for common artifacts
        if "error" in output.lower():
            artifacts.append("error_mention")
        if "undefined" in output.lower():
            artifacts.append("undefined_reference")
        if output.count("(") != output.count(")"):
            artifacts.append("unbalanced_parentheses")
        
        return artifacts
    
    def _generate_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate improvement recommendations based on test results"""
        recommendations = []
        
        # Analyze test results by type
        by_type = {}
        for result in results:
            if result.test_type not in by_type:
                by_type[result.test_type] = []
            by_type[result.test_type].append(result.score)
        
        # Generate specific recommendations
        for test_type, scores in by_type.items():
            avg_score = statistics.mean(scores)
            
            if test_type == TestType.CONSISTENCY and avg_score < 0.7:
                recommendations.append("Improve consistency: Add stronger operational anchors")
            
            elif test_type == TestType.DISTINCTIVENESS and avg_score < 0.6:
                recommendations.append("Enhance distinctiveness: Use more specific linguistic materials")
            
            elif test_type == TestType.RESISTANCE and avg_score < 0.6:
                recommendations.append("Increase resistance: Strengthen core operational structure")
            
            elif test_type == TestType.PURITY and avg_score < 0.7:
                recommendations.append("Improve purity: Remove ambiguous or generic elements")
            
            elif test_type == TestType.PERFORMANCE and avg_score < 0.5:
                recommendations.append("Optimize performance: Simplify complex structures")
        
        if not recommendations:
            recommendations.append("Prompt meets quality standards - consider advanced refinement")
        
        return recommendations
    
    def _print_report_summary(self, report: ValidationReport) -> None:
        """Print a summary of the validation report"""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {report.prompt_id}")
        print(f"{'='*60}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Score: {report.overall_score:.2%}")
        print(f"Status: {'✓ PASSED' if report.passed else '✗ FAILED'}")
        
        # Score breakdown
        print(f"\nScore Breakdown:")
        by_type = {}
        for result in report.test_results:
            if result.test_type not in by_type:
                by_type[result.test_type] = []
            by_type[result.test_type].append(result.score)
        
        for test_type, scores in by_type.items():
            avg_score = statistics.mean(scores)
            status = "✓" if avg_score > 0.6 else "✗"
            print(f"  {status} {test_type.value}: {avg_score:.2%}")
        
        # Recommendations
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  → {rec}")
        
        print(f"{'='*60}\n")
    
    def save_report(self, report: ValidationReport, filepath: Optional[Path] = None) -> None:
        """Save validation report to file"""
        if filepath is None:
            filepath = self.base_path / f"validation_report_{report.prompt_id}.json"
        
        report_dict = {
            "prompt_id": report.prompt_id,
            "prompt_content": report.prompt_content,
            "overall_score": report.overall_score,
            "passed": report.passed,
            "recommendations": report.recommendations,
            "timestamp": report.timestamp,
            "test_summary": {
                test_type.value: {
                    "scores": [r.score for r in report.test_results if r.test_type == test_type],
                    "avg_score": statistics.mean([r.score for r in report.test_results if r.test_type == test_type])
                    if [r.score for r in report.test_results if r.test_type == test_type] else 0.0
                }
                for test_type in TestType
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Report saved to {filepath}")


def main():
    """Main execution for testing suite demonstration"""
    print("═" * 60)
    print("PROMPTKRAFT TESTING SUITE")
    print("Rigorous Validation of Operational Prompts")
    print("═" * 60)
    
    try:
        # Initialize test harness (requires API keys)
        harness = PromptTestHarness(base_path=Path("."))
        
        print(f"\nUsing LLM Provider: {harness.llm_config.provider.value}")
        print(f"Model: {harness.llm_config.model}")
        print("-" * 60)
    except ValueError as e:
        print(f"\n⚠ Error: {e}")
        print("\nPlease set up your API keys in the .env file:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  ANTHROPIC_API_KEY=your-key-here")
        return
    
    # Test prompts
    test_prompts = [
        "≡ I am careful scribing ≡ What appears before me: {input}",
        "? I am seeking-into ? From what I observe: {data} What questions emerge?",
        "! I am giving-form-to-insight ! From these questions: {questions} What crystallizes?",
        "⇔ I am bending-back-to-examine ⇔ Examining this insight: {insight}",
    ]
    
    # Run validation on each prompt
    for prompt in test_prompts:
        report = harness.run_full_validation(prompt)
        harness.save_report(report)
        
        # Brief pause between tests
        time.sleep(0.5)
    
    print("\n" + "═" * 60)
    print("Testing session complete")
    print(f"Total prompts tested: {len(test_prompts)}")
    print(f"Reports generated: {len(harness.test_history)}")
    print("═" * 60)


if __name__ == "__main__":
    main()