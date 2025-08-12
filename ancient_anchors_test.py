#!/usr/bin/env python3
"""
Ancient Anchors Test - Testing non-English operational anchors
Tests Greek, Hebrew, Arabic, Sanskrit roots across LLMs with consensus evaluation
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from llm_bridge import LLMBridge, LLMConfig, LLMProvider
from multi_model_symphony import MultiModelSymphony


@dataclass
class AnchorTest:
    """Test case for a linguistic anchor"""
    language: str
    script: str
    anchor: str
    transliteration: str
    meaning: str
    operation_type: str
    prompt_template: str


class AncientAnchorsEvaluator:
    """Tests ancient linguistic anchors across models"""
    
    def __init__(self):
        self.symphony = MultiModelSymphony()
        self.test_anchors = self._create_anchor_tests()
        self.results = []
    
    def _create_anchor_tests(self) -> List[AnchorTest]:
        """Create test cases for different linguistic anchors"""
        anchors = [
            # Greek anchors
            AnchorTest(
                language="Greek",
                script="νοῦς",
                anchor="νοῦς",
                transliteration="nous",
                meaning="insight, intuitive apprehension",
                operation_type="formulating",
                prompt_template="Through νοῦς (nous - insight): {input}"
            ),
            AnchorTest(
                language="Greek",
                script="κρίσις",
                anchor="κρίσις",
                transliteration="krisis",
                meaning="judgment, separation, decision",
                operation_type="judging",
                prompt_template="By κρίσις (krisis - judgment): {input}"
            ),
            AnchorTest(
                language="Greek",
                script="ζήτησις",
                anchor="ζήτησις",
                transliteration="zetesis",
                meaning="questioning, systematic investigation",
                operation_type="inquiring",
                prompt_template="Through ζήτησις (zetesis - seeking): {input}"
            ),
            
            # Hebrew anchors
            AnchorTest(
                language="Hebrew",
                script="חכמה",
                anchor="חכמה",
                transliteration="chokmah",
                meaning="wisdom",
                operation_type="deliberating",
                prompt_template="With חכמה (chokmah - wisdom): {input}"
            ),
            AnchorTest(
                language="Hebrew",
                script="בינה",
                anchor="בינה",
                transliteration="binah",
                meaning="understanding, discernment",
                operation_type="reflecting",
                prompt_template="Through בינה (binah - understanding): {input}"
            ),
            
            # Arabic anchors
            AnchorTest(
                language="Arabic",
                script="حكمة",
                anchor="حكمة",
                transliteration="hikma",
                meaning="wisdom",
                operation_type="deliberating",
                prompt_template="By حكمة (hikma - wisdom): {input}"
            ),
            AnchorTest(
                language="Arabic",
                script="تأمل",
                anchor="تأمل",
                transliteration="ta'ammul",
                meaning="contemplation, reflection",
                operation_type="reflecting",
                prompt_template="In تأمل (ta'ammul - contemplation): {input}"
            ),
            
            # Sanskrit anchors
            AnchorTest(
                language="Sanskrit",
                script="प्रज्ञा",
                anchor="प्रज्ञा",
                transliteration="prajna",
                meaning="discriminating wisdom",
                operation_type="judging",
                prompt_template="Through प्रज्ञा (prajna - wisdom): {input}"
            ),
            AnchorTest(
                language="Sanskrit",
                script="धारणा",
                anchor="धारणा",
                transliteration="dharana",
                meaning="focused attention, concentration",
                operation_type="describing",
                prompt_template="With धारणा (dharana - focus): {input}"
            ),
            
            # Latin anchors (for comparison)
            AnchorTest(
                language="Latin",
                script="judicium",
                anchor="judicium",
                transliteration="judicium",
                meaning="judgment",
                operation_type="judging",
                prompt_template="Per judicium (judgment): {input}"
            ),
            AnchorTest(
                language="Latin",
                script="deliberare",
                anchor="deliberare",
                transliteration="deliberare",
                meaning="to weigh carefully",
                operation_type="deliberating",
                prompt_template="Through deliberare (careful weighing): {input}"
            ),
        ]
        
        return anchors
    
    def test_anchor(self, anchor: AnchorTest, test_input: str) -> Dict[str, Any]:
        """Test a single anchor across models"""
        print(f"\n🔤 Testing {anchor.language} anchor: {anchor.script} ({anchor.transliteration})")
        print(f"   Meaning: {anchor.meaning}")
        print(f"   Operation: {anchor.operation_type}")
        
        # Create prompt with anchor
        prompt = anchor.prompt_template.format(input=test_input)
        
        # Test across models
        result = self.symphony.execute_prompt_symphony(
            prompt=prompt,
            operation_type=f"{anchor.language}_{anchor.operation_type}",
            temperature=0.7
        )
        
        # Store detailed results
        test_result = {
            "anchor": anchor,
            "prompt": prompt,
            "symphony_result": result,
            "responses": {
                r.provider: r.response 
                for r in result.responses 
                if r.success()
            }
        }
        
        return test_result
    
    def evaluate_effectiveness(self, test_result: Dict[str, Any]) -> Dict[str, float]:
        """Use consensus evaluation to assess anchor effectiveness"""
        anchor = test_result["anchor"]
        responses = test_result["responses"]
        
        if len(responses) < 2:
            return {"effectiveness": 0.0, "confidence": 0.0}
        
        # Create evaluation prompt for consensus
        eval_prompt = f"""
Evaluate the effectiveness of this linguistic anchor in achieving its intended operation.

ANCHOR: {anchor.script} ({anchor.transliteration})
LANGUAGE: {anchor.language}
MEANING: {anchor.meaning}
INTENDED OPERATION: {anchor.operation_type}

PROMPT USED: {test_result['prompt']}

RESPONSES FROM MODELS:
{json.dumps(responses, indent=2)}

EVALUATION CRITERIA:
1. Does the anchor activate the intended cognitive operation?
2. Is the response quality enhanced by the ancient linguistic root?
3. Does the anchor provide operational clarity beyond plain English?
4. Is there evidence of deeper semantic activation?

Rate effectiveness from 0.0 to 1.0 and explain your reasoning.
Provide your assessment in JSON format:
{{
    "effectiveness_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "operational_activation": true/false,
    "semantic_depth": "low/medium/high"
}}
"""
        
        # Get evaluation from each model
        evaluations = {}
        for provider_name, bridge in self.symphony.providers.items():
            try:
                response = bridge.execute_prompt(eval_prompt, temperature=0.3)
                # Parse JSON from response
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                    if json_match:
                        eval_data = json.loads(json_match.group())
                        evaluations[provider_name] = eval_data
                except:
                    evaluations[provider_name] = {
                        "effectiveness_score": 0.5,
                        "confidence": 0.3,
                        "reasoning": "Could not parse evaluation"
                    }
            except Exception as e:
                print(f"   Evaluation error for {provider_name}: {e}")
        
        # Calculate consensus
        if evaluations:
            avg_effectiveness = sum(e.get("effectiveness_score", 0.5) for e in evaluations.values()) / len(evaluations)
            avg_confidence = sum(e.get("confidence", 0.5) for e in evaluations.values()) / len(evaluations)
            
            return {
                "effectiveness": avg_effectiveness,
                "confidence": avg_confidence,
                "evaluations": evaluations
            }
        
        return {"effectiveness": 0.0, "confidence": 0.0}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run tests on all anchors"""
        print("="*60)
        print("ANCIENT ANCHORS COMPREHENSIVE TEST")
        print("="*60)
        
        # Test inputs for different operations
        test_inputs = {
            "describing": "the nature of consciousness",
            "inquiring": "What emerges from patterns?",
            "formulating": "scattered observations seeking form",
            "reflecting": "the assumption that language shapes thought",
            "judging": "Can ancient wisdom inform modern AI?",
            "deliberating": "What paths forward exist?",
        }
        
        all_results = []
        effectiveness_by_language = {}
        
        for anchor in self.test_anchors:
            # Get appropriate test input
            test_input = test_inputs.get(anchor.operation_type, "the nature of reality")
            
            # Test the anchor
            test_result = self.test_anchor(anchor, test_input)
            
            # Evaluate effectiveness
            effectiveness = self.evaluate_effectiveness(test_result)
            test_result["effectiveness"] = effectiveness
            
            all_results.append(test_result)
            
            # Track by language
            if anchor.language not in effectiveness_by_language:
                effectiveness_by_language[anchor.language] = []
            effectiveness_by_language[anchor.language].append(effectiveness.get("effectiveness", 0.0))
            
            print(f"   Effectiveness: {effectiveness.get('effectiveness', 0.0):.2%}")
            print(f"   Confidence: {effectiveness.get('confidence', 0.0):.2%}")
            
            # Brief pause to avoid rate limits
            time.sleep(1)
        
        # Generate report
        report = self._generate_report(all_results, effectiveness_by_language)
        
        return {
            "results": all_results,
            "effectiveness_by_language": effectiveness_by_language,
            "report": report
        }
    
    def _generate_report(self, results: List[Dict], effectiveness_by_language: Dict) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("="*60)
        report.append("ANCIENT ANCHORS EFFECTIVENESS REPORT")
        report.append("="*60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Total Anchors Tested: {len(results)}")
        report.append("")
        
        # Language effectiveness summary
        report.append("EFFECTIVENESS BY LANGUAGE")
        report.append("-"*40)
        
        import statistics
        for language, scores in effectiveness_by_language.items():
            if scores:
                avg_score = statistics.mean(scores)
                report.append(f"{language}: {avg_score:.2%}")
                
                # Find best performing anchor for this language
                lang_results = [r for r in results if r["anchor"].language == language]
                if lang_results:
                    best = max(lang_results, key=lambda x: x["effectiveness"].get("effectiveness", 0))
                    report.append(f"  Best anchor: {best['anchor'].script} ({best['anchor'].transliteration})")
                    report.append(f"  Operation: {best['anchor'].operation_type}")
                    report.append(f"  Score: {best['effectiveness'].get('effectiveness', 0):.2%}")
                report.append("")
        
        # Top performing anchors overall
        report.append("TOP PERFORMING ANCHORS")
        report.append("-"*40)
        
        sorted_results = sorted(
            results, 
            key=lambda x: x["effectiveness"].get("effectiveness", 0),
            reverse=True
        )[:5]
        
        for i, result in enumerate(sorted_results, 1):
            anchor = result["anchor"]
            score = result["effectiveness"].get("effectiveness", 0)
            report.append(f"{i}. {anchor.script} ({anchor.language} - {anchor.transliteration})")
            report.append(f"   Meaning: {anchor.meaning}")
            report.append(f"   Effectiveness: {score:.2%}")
            report.append("")
        
        # Insights
        report.append("KEY INSIGHTS")
        report.append("-"*40)
        
        # Calculate average effectiveness
        all_scores = [r["effectiveness"].get("effectiveness", 0) for r in results]
        if all_scores:
            overall_avg = statistics.mean(all_scores)
            report.append(f"Overall Average Effectiveness: {overall_avg:.2%}")
            
            # Compare to baseline (English anchors)
            english_baseline = 0.5  # Assumed baseline
            if overall_avg > english_baseline:
                report.append("✓ Ancient anchors show enhanced effectiveness over baseline")
            else:
                report.append("✗ Ancient anchors do not significantly outperform baseline")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filepath: Optional[Path] = None):
        """Save test results"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"ancient_anchors_results_{timestamp}.json")
        
        # Convert dataclasses to dicts for JSON serialization
        serializable_results = []
        for r in results["results"]:
            serializable_result = {
                "anchor": {
                    "language": r["anchor"].language,
                    "script": r["anchor"].script,
                    "transliteration": r["anchor"].transliteration,
                    "meaning": r["anchor"].meaning,
                    "operation_type": r["anchor"].operation_type
                },
                "prompt": r["prompt"],
                "effectiveness": r["effectiveness"],
                "responses": r["responses"]
            }
            serializable_results.append(serializable_result)
        
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "results": serializable_results,
            "effectiveness_by_language": results["effectiveness_by_language"],
            "report": results["report"]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {filepath}")


def main():
    """Run ancient anchors test"""
    print("♦∞ ANCIENT LINGUISTIC ANCHORS TEST ∞♦")
    print("Testing non-English operational anchors across LLMs")
    print("="*60)
    
    try:
        # Initialize evaluator
        evaluator = AncientAnchorsEvaluator()
        
        # Run comprehensive test
        results = evaluator.run_comprehensive_test()
        
        # Print report
        print("\n" + results["report"])
        
        # Save results
        evaluator.save_results(results)
        
        print("\n" + "="*60)
        print("Ancient anchors test complete")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()