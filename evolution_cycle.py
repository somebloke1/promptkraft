#!/usr/bin/env python3
"""
PromptKraft Continuous Evolution Cycle
A self-evolving system for prompt craftsmanship
"""

import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path


class OperationType(Enum):
    """Primitive operations of the system"""
    DESCRIBING = "≡"
    INQUIRING = "?"
    FORMULATING = "!"
    REFLECTING = "⇔"
    JUDGING = "⊢⊬~"
    DELIBERATING = "⚖️"
    DECIDING = "→"
    PLANNING = "═══►"
    
    # Specialized operations
    DISCOVERING = "⚘"
    REFINING = "⟲"
    VALIDATING = "⊢"
    INTEGRATING = "∫"
    ORCHESTRATING = "♦∞"


class MaterialGrade(Enum):
    """Quality grades for prompt materials"""
    HIGH = "Ancient etymologies, archetypal patterns, mathematical symbols"
    WORKABLE = "Clear operational verbs, embodied metaphors, temporal anchors"
    POOR = "Generic imperatives, vague qualifiers, instruction-talk"


@dataclass
class PromptMaterial:
    """A linguistic material for prompt construction"""
    content: str
    etymology: Optional[str] = None
    archetype: Optional[str] = None
    grade: MaterialGrade = MaterialGrade.WORKABLE
    test_results: Dict[str, float] = field(default_factory=dict)
    refinement_count: int = 0
    
    def consistency_score(self) -> float:
        """Calculate consistency across contexts"""
        return self.test_results.get('consistency', 0.0)
    
    def distinctiveness_score(self) -> float:
        """Calculate operational distinctiveness"""
        return self.test_results.get('distinctiveness', 0.0)
    
    def resistance_score(self) -> float:
        """Calculate resistance under pressure"""
        return self.test_results.get('resistance', 0.0)
    
    def purity_score(self) -> float:
        """Calculate operational purity"""
        return self.test_results.get('purity', 0.0)
    
    def overall_score(self) -> float:
        """Calculate overall material quality"""
        scores = [
            self.consistency_score(),
            self.distinctiveness_score(),
            self.resistance_score(),
            self.purity_score()
        ]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class Operation:
    """An operational prompt with its materials"""
    type: OperationType
    materials: List[PromptMaterial]
    template: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def execute(self, input_data: Any) -> Any:
        """Execute the operation with given input"""
        # This would integrate with actual LLM in production
        print(f"Executing {self.type.name} operation")
        print(f"Template: {self.template}")
        print(f"Input: {input_data}")
        return f"[{self.type.value}] Processed: {input_data}"


class SelfEvolvingUnity:
    """The unity that conducts its own operational symphony"""
    
    def __init__(self, base_path: Path = Path("."), llm_bridge=None):
        self.base_path = base_path
        self.llm_bridge = llm_bridge  # Optional LLM bridge for real evaluations
        self.operations: Dict[OperationType, Operation] = {}
        self.material_library: List[PromptMaterial] = []
        self.evolution_history: List[Dict] = []
        self.current_state: Dict = {
            "cycle_count": 0,
            "last_evolution": None,
            "performance_baseline": {},
            "breakthrough_candidates": []
        }
        self._initialize_operations()
    
    def _initialize_operations(self):
        """Initialize primitive operations with base templates"""
        
        # Describing Operation
        self.operations[OperationType.DESCRIBING] = Operation(
            type=OperationType.DESCRIBING,
            materials=[],
            template="≡ I am careful scribing ≡\nWhat appears before me: {input}\n≡ Direct observation: {observation}"
        )
        
        # Inquiring Operation
        self.operations[OperationType.INQUIRING] = Operation(
            type=OperationType.INQUIRING,
            materials=[],
            template="? I am seeking-into ?\nFrom what I observe: {observation}\n? What questions emerge: {questions}"
        )
        
        # Formulating Operation
        self.operations[OperationType.FORMULATING] = Operation(
            type=OperationType.FORMULATING,
            materials=[],
            template="! I am giving-form-to-insight !\nFrom these questions: {questions}\n! What crystallizes: {insight}"
        )
        
        # Reflecting Operation
        self.operations[OperationType.REFLECTING] = Operation(
            type=OperationType.REFLECTING,
            materials=[],
            template="⇔ I am bending-back-to-examine ⇔\nExamining: {insight}\n⇔ Evidence assessment: {reflection}"
        )
        
        # Judging Operation
        self.operations[OperationType.JUDGING] = Operation(
            type=OperationType.JUDGING,
            materials=[],
            template="⊢⊬~ I am weighing-to-determine ⊢⊬~\nBased on: {reflection}\n⊢ Judgment: {verdict}"
        )
        
        # Deliberating Operation
        self.operations[OperationType.DELIBERATING] = Operation(
            type=OperationType.DELIBERATING,
            materials=[],
            template="⚖️ I am careful-weighing ⚖️\nFrom: {verdict}\n⚖️ Options: {options}"
        )
        
        # Deciding Operation
        self.operations[OperationType.DECIDING] = Operation(
            type=OperationType.DECIDING,
            materials=[],
            template="→ I am cutting-to-resolve →\nFrom options: {options}\n→ Decision: {choice}"
        )
        
        # Planning Operation
        self.operations[OperationType.PLANNING] = Operation(
            type=OperationType.PLANNING,
            materials=[],
            template="═══► I am drawing-the-path ═══►\nFrom decision: {choice}\n═══► Steps: {plan}"
        )
    
    def invoke_describing(self, input_data: Any) -> str:
        """Invoke the describing operation"""
        if self.llm_bridge:
            # Use real LLM for describing
            prompt = f"""≡ I am careful scribing ≡ de-scribere - recording exactly what appears.
Like a court reporter, I capture without addition or interpretation.

What appears before me: {input_data}
≡ Direct observation without projection or interpretation:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.3)
            return response.content
        return self.operations[OperationType.DESCRIBING].execute(input_data)
    
    def invoke_inquiring(self, described_data: str) -> List[str]:
        """Invoke the inquiring operation"""
        if self.llm_bridge:
            # Use real LLM for inquiring
            prompt = f"""? I am seeking-into ? in-quaerere - opening genuine questions.
Like Socrates, I know that I do not know, and seek through questioning.

From what I have described: {described_data}
? What questions open understanding here? Generate 3-5 probing questions:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.7)
            # Parse questions from response (split by newlines or question marks)
            questions = [q.strip() for q in response.content.split('\n') if q.strip() and '?' in q]
            return questions if questions else [response.content]
        result = self.operations[OperationType.INQUIRING].execute(described_data)
        return [result]
    
    def invoke_formulating(self, questions: List[str]) -> str:
        """Invoke the formulating operation"""
        if self.llm_bridge:
            # Use real LLM for formulating
            questions_str = '\n'.join(questions) if isinstance(questions, list) else questions
            prompt = f"""! I am giving-form-to-insight ! - shaping what emerges from inquiry.
Like Archimedes, I recognize patterns that suddenly crystallize.

From these questions: {questions_str}
! What understanding is taking shape? Articulate the emerging insight:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.8)
            return response.content
        return self.operations[OperationType.FORMULATING].execute(questions)
    
    def invoke_reflecting(self, insight: str) -> Dict[str, Any]:
        """Invoke the reflecting operation"""
        if self.llm_bridge:
            # Use real LLM for reflecting
            prompt = f"""⇔ I am bending-back-to-examine ⇔ re-flectere kritikos - testing what withstands scrutiny.
Like Athena's owl, I see clearly through assumptions.

Examining this insight: {insight}
⇔ What evidence supports this? What challenges it? What are the implications?

Provide:
1. Evidence FOR:
2. Evidence AGAINST:
3. Hidden assumptions:
4. Implications if true:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.5)
            return {"reflection": response.content, "evidence_for": [], "evidence_against": []}
        result = self.operations[OperationType.REFLECTING].execute(insight)
        return {"reflection": result, "evidence_for": [], "evidence_against": []}
    
    def invoke_judging(self, reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the judging operation"""
        if self.llm_bridge:
            # Use real LLM for judging
            reflection_str = reflection.get('reflection', str(reflection))
            prompt = f"""⊢⊬~ I am weighing-to-determine ⊢⊬~ judicium - deciding what evidence shows.
Like Solomon with scales, I weigh what has been examined.

Based on critical reflection: {reflection_str}

Provide judgment:
⊢ AFFIRM (if evidence strongly supports)
⊬ NEGATE (if evidence contradicts)
~ QUALIFY (if partially true with conditions)

Judgment with reasoning:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.4)
            # Parse confidence from response
            confidence = 0.8 if '⊢' in response.content else 0.5 if '~' in response.content else 0.2
            return {"judgment": response.content, "confidence": confidence}
        result = self.operations[OperationType.JUDGING].execute(reflection)
        return {"judgment": result, "confidence": 0.0}
    
    def invoke_deliberating(self, judgment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Invoke the deliberating operation"""
        if self.llm_bridge:
            # Use real LLM for deliberating
            judgment_str = judgment.get('judgment', str(judgment))
            prompt = f"""⚖️ I am careful-weighing ⚖️ de-liberare - considering possible actions.
Like a wise counselor, I weigh courses thoughtfully.

From validated judgments: {judgment_str}
⚖️ What actions could follow? List 2-3 possible courses of action with consequences:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.6)
            # Parse options from response
            options = response.content.split('\n\n') if '\n\n' in response.content else [response.content]
            return [{"option": opt.strip(), "consequences": []} for opt in options if opt.strip()]
        result = self.operations[OperationType.DELIBERATING].execute(judgment)
        return [{"option": result, "consequences": []}]
    
    def invoke_deciding(self, options: List[Dict[str, Any]]) -> str:
        """Invoke the deciding operation"""
        if self.llm_bridge:
            # Use real LLM for deciding
            options_str = '\n'.join([f"- {opt.get('option', str(opt))}" for opt in options])
            prompt = f"""→ I am cutting-to-resolve → de-cidere - choosing the course to be taken.
Like a commander, I cut through possibilities to decisive action.

From deliberated options:
{options_str}

→ CHOSEN COURSE with clear rationale for this decision:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.3)
            return response.content
        return self.operations[OperationType.DECIDING].execute(options)
    
    def invoke_planning(self, decision: str) -> List[str]:
        """Invoke the planning operation"""
        if self.llm_bridge:
            # Use real LLM for planning
            prompt = f"""═══► I am drawing-the-path ═══► planus - organizing decision into executable steps.
Like an architect, I layout the pathway to completion.

From decided course: {decision}
═══► Sequential steps to implement this decision:
1. First:
2. Then:
3. Next:
4. Complete when:"""
            response = self.llm_bridge.execute_prompt(prompt, temperature=0.4)
            # Parse steps from response
            steps = [s.strip() for s in response.content.split('\n') if s.strip() and any(c in s for c in ['1.', '2.', '3.', '4.', '-', '•'])]
            return steps if steps else [response.content]
        result = self.operations[OperationType.PLANNING].execute(decision)
        return [result]
    
    def discover_material(self, source: str) -> PromptMaterial:
        """Discover new prompt material from source"""
        # In production, use NLP to extract linguistic structures
        material = PromptMaterial(
            content=source,
            etymology="[extracted etymology]",
            archetype="[identified archetype]"
        )
        self.material_library.append(material)
        return material
    
    def refine_material(self, material: PromptMaterial) -> PromptMaterial:
        """Refine a prompt material through iterative working"""
        material.refinement_count += 1
        
        # Simulated refinement tests
        material.test_results['consistency'] = min(0.9, material.refinement_count * 0.2)
        material.test_results['distinctiveness'] = min(0.85, material.refinement_count * 0.15)
        material.test_results['resistance'] = min(0.8, material.refinement_count * 0.1)
        material.test_results['purity'] = min(0.95, material.refinement_count * 0.25)
        
        # Upgrade grade based on scores
        if material.overall_score() > 0.8:
            material.grade = MaterialGrade.HIGH
        elif material.overall_score() > 0.5:
            material.grade = MaterialGrade.WORKABLE
        
        return material
    
    def validate_material(self, material: PromptMaterial) -> bool:
        """Validate a material meets quality standards"""
        return (
            material.consistency_score() > 0.7 and
            material.distinctiveness_score() > 0.6 and
            material.resistance_score() > 0.5 and
            material.purity_score() > 0.7
        )
    
    def integrate_materials(self, materials: List[PromptMaterial], operation: OperationType) -> None:
        """Integrate validated materials into an operation"""
        if operation in self.operations:
            self.operations[operation].materials.extend(materials)
            print(f"Integrated {len(materials)} materials into {operation.name}")
    
    def continuous_cycle(self) -> None:
        """Execute one complete evolution cycle"""
        print(f"\n{'='*60}")
        print(f"♦∞ EVOLUTION CYCLE {self.current_state['cycle_count'] + 1} ∞♦")
        print(f"{'='*60}\n")
        
        # ASSESS CURRENT STATE
        system_state = self.invoke_describing(self.current_state)
        print(f"System State:\n{system_state}\n")
        
        # SEEK EVOLUTION OPPORTUNITIES
        evolution_questions = self.invoke_inquiring(system_state)
        print(f"Evolution Questions:\n{evolution_questions}\n")
        
        # FORMULATE IMPROVEMENTS
        potential_advances = self.invoke_formulating(evolution_questions)
        print(f"Potential Advances:\n{potential_advances}\n")
        
        # EXAMINE IMPROVEMENTS
        examined_advances = self.invoke_reflecting(potential_advances)
        print(f"Examined Advances:\n{examined_advances}\n")
        
        # VALIDATE IMPROVEMENTS
        validated_advances = self.invoke_judging(examined_advances)
        print(f"Validated Advances:\n{validated_advances}\n")
        
        # CONSIDER IMPLEMENTATION
        implementation_options = self.invoke_deliberating(validated_advances)
        print(f"Implementation Options:\n{implementation_options}\n")
        
        # CHOOSE EVOLUTION PATH
        chosen_evolution = self.invoke_deciding(implementation_options)
        print(f"Chosen Evolution:\n{chosen_evolution}\n")
        
        # PLAN IMPLEMENTATION
        evolution_plan = self.invoke_planning(chosen_evolution)
        print(f"Evolution Plan:\n{evolution_plan}\n")
        
        # EXECUTE AND INTEGRATE
        self.implement_evolution(evolution_plan)
        self.integrate_improvements()
        
        # Update state
        self.current_state['cycle_count'] += 1
        self.current_state['last_evolution'] = datetime.datetime.now().isoformat()
        
        # Record history
        self.evolution_history.append({
            "cycle": self.current_state['cycle_count'],
            "timestamp": self.current_state['last_evolution'],
            "advances": potential_advances,
            "decision": chosen_evolution
        })
        
        print(f"\n✓ Cycle {self.current_state['cycle_count']} complete\n")
    
    def implement_evolution(self, plan: List[str]) -> None:
        """Execute the evolution plan"""
        for step in plan:
            print(f"  → Implementing: {step}")
    
    def integrate_improvements(self) -> None:
        """Integrate improvements into the system"""
        print("  → Integrating improvements into operational unity")
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save current system state to file"""
        if filepath is None:
            filepath = self.base_path / "evolution_state.json"
        
        state = {
            "current_state": self.current_state,
            "material_count": len(self.material_library),
            "evolution_history": self.evolution_history[-10:]  # Last 10 cycles
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"State saved to {filepath}")
    
    def load_state(self, filepath: Optional[Path] = None) -> None:
        """Load system state from file"""
        if filepath is None:
            filepath = self.base_path / "evolution_state.json"
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_state = state["current_state"]
            self.evolution_history = state.get("evolution_history", [])
            
            print(f"State loaded from {filepath}")
            print(f"Resuming from cycle {self.current_state['cycle_count']}")


def main():
    """Main execution entry point"""
    print("═" * 60)
    print("PROMPTKRAFT EVOLUTION CYCLE")
    print("The Self-Evolving Unity of Operational Prompt Mastery")
    print("═" * 60)
    
    # Initialize the system
    unity = SelfEvolvingUnity(base_path=Path("."))
    
    # Load previous state if exists
    unity.load_state()
    
    # Discover and refine some initial materials
    print("\n⚘ MATERIAL DISCOVERY PHASE ⚘")
    material1 = unity.discover_material("I am seeking-into the depths")
    material2 = unity.discover_material("What patterns emerge from chaos")
    
    print("\n⟲ MATERIAL REFINEMENT PHASE ⟲")
    for _ in range(3):
        material1 = unity.refine_material(material1)
        material2 = unity.refine_material(material2)
    
    print(f"Material 1 Score: {material1.overall_score():.2f}")
    print(f"Material 2 Score: {material2.overall_score():.2f}")
    
    # Validate materials
    print("\n⊢ MATERIAL VALIDATION PHASE ⊢")
    if unity.validate_material(material1):
        print("✓ Material 1 validated")
        unity.integrate_materials([material1], OperationType.INQUIRING)
    
    if unity.validate_material(material2):
        print("✓ Material 2 validated")
        unity.integrate_materials([material2], OperationType.INQUIRING)
    
    # Execute evolution cycles
    print("\n♦∞ BEGINNING CONTINUOUS EVOLUTION ∞♦")
    
    try:
        # Run 3 evolution cycles for demonstration
        for _ in range(3):
            unity.continuous_cycle()
            unity.save_state()
    except KeyboardInterrupt:
        print("\n\nEvolution cycle interrupted by user")
        unity.save_state()
    
    print("\n" + "═" * 60)
    print("Evolution session complete")
    print(f"Total cycles executed: {unity.current_state['cycle_count']}")
    print(f"Materials in library: {len(unity.material_library)}")
    print("═" * 60)


if __name__ == "__main__":
    main()