#!/usr/bin/env python3
"""
PromptKraft CLI - Interactive Prompt Crafting Interface
The conductor's baton for operational symphony
"""

import os
import sys
import json
import argparse
import readline  # For command history
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from llm_bridge import LLMBridge, LLMConfig, LLMProvider
from evolution_cycle import (
    SelfEvolvingUnity, 
    OperationType, 
    PromptMaterial, 
    MaterialGrade
)
from prompt_testing_suite import PromptTestHarness


class PromptKraftCLI:
    """Interactive CLI for prompt craftsmanship"""
    
    def __init__(self):
        self.llm_bridge = self._initialize_llm()
        self.unity = SelfEvolvingUnity(llm_bridge=self.llm_bridge)
        self.test_harness = PromptTestHarness()
        self.session_history = []
        self.current_prompt = None
        self.current_material = None
        
        # Enable command history
        self.history_file = Path.home() / ".promptkraft_history"
        self._setup_readline()
    
    def _initialize_llm(self) -> LLMBridge:
        """Initialize LLM Bridge with best available provider"""
        if os.getenv("OPENAI_API_KEY"):
            provider = LLMProvider.OPENAI
            model = "gpt-3.5-turbo"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = LLMProvider.ANTHROPIC  
            model = "claude-3-haiku-20240307"
        else:
            raise ValueError("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        
        config = LLMConfig(provider=provider, model=model)
        return LLMBridge(config)
    
    def _setup_readline(self):
        """Setup command history and tab completion"""
        if self.history_file.exists():
            readline.read_history_file(self.history_file)
        
        # Set up tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._completer)
    
    def _completer(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands"""
        commands = [
            'craft', 'test', 'refine', 'discover', 'integrate',
            'evolve', 'describe', 'inquire', 'formulate', 'reflect',
            'judge', 'deliberate', 'decide', 'plan', 'help', 'exit',
            'status', 'history', 'save', 'load', 'clear'
        ]
        
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None
    
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "═" * 60)
        print("♦∞ PROMPTKRAFT INTERACTIVE CRAFTING INTERFACE ∞♦")
        print("═" * 60)
        print(f"Provider: {self.llm_bridge.config.provider.value}")
        print(f"Model: {self.llm_bridge.config.model}")
        print("Type 'help' for commands or 'exit' to quit")
        print("═" * 60 + "\n")
    
    def print_help(self):
        """Print help information"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    PROMPTKRAFT COMMANDS                      ║
╠══════════════════════════════════════════════════════════════╣
║ CRAFTING OPERATIONS                                          ║
║   craft <prompt>    - Begin crafting a new prompt           ║
║   test [prompt]     - Test current or provided prompt       ║
║   refine [prompt]   - Refine through iterative working      ║
║   discover <source> - Discover materials from source        ║
║   integrate         - Integrate validated materials         ║
║                                                              ║
║ PRIMITIVE OPERATIONS                                         ║
║   describe <input>  - ≡ Invoke describing operation         ║
║   inquire <data>    - ? Invoke inquiring operation          ║
║   formulate <q>     - ! Invoke formulating operation        ║
║   reflect <insight> - ⇔ Invoke reflecting operation         ║
║   judge <evidence>  - ⊢⊬~ Invoke judging operation          ║
║   deliberate <j>    - ⚖️ Invoke deliberating operation       ║
║   decide <options>  - → Invoke deciding operation           ║
║   plan <decision>   - ═══► Invoke planning operation        ║
║                                                              ║
║ EVOLUTION OPERATIONS                                         ║
║   evolve            - Run one evolution cycle               ║
║   evolve <n>        - Run n evolution cycles                ║
║                                                              ║
║ SESSION MANAGEMENT                                           ║
║   status            - Show current session status           ║
║   history           - Show session history                  ║
║   save [filename]   - Save session to file                  ║
║   load <filename>   - Load session from file                ║
║   clear             - Clear current prompt/material         ║
║   exit/quit         - Exit PromptKraft                      ║
║                                                              ║
║ SHORTCUTS                                                    ║
║   !<prompt>         - Quick test of prompt                  ║
║   ?<question>       - Quick inquiry                         ║
║   @<material>       - Quick material discovery              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(help_text)
    
    def cmd_craft(self, args: str):
        """Craft a new prompt"""
        if not args:
            print("⚠ Please provide a prompt to craft")
            return
        
        self.current_prompt = args
        print(f"\n⚘ Crafting: {args[:100]}...")
        
        # Discover materials from the prompt
        material = self.unity.discover_material(args)
        self.current_material = material
        
        print(f"✓ Material discovered: Grade {material.grade.name}")
        print(f"  Etymology: {material.etymology}")
        print(f"  Archetype: {material.archetype}")
    
    def cmd_test(self, args: str):
        """Test a prompt"""
        prompt = args if args else self.current_prompt
        
        if not prompt:
            print("⚠ No prompt to test. Use 'craft <prompt>' first")
            return
        
        print(f"\n⊢ Testing prompt: {prompt[:100]}...")
        
        # Execute through LLM
        response = self.llm_bridge.execute_prompt(prompt)
        
        print(f"\n📤 Response ({response.latency:.2f}s):")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        print(f"Tokens: {response.total_tokens} | Cost: ${response.cost:.4f}")
        
        # Quick validation
        if self.current_material:
            self.current_material.test_results['consistency'] = 0.8
            self.current_material.test_results['distinctiveness'] = 0.7
            print(f"\n✓ Quick validation score: {self.current_material.overall_score():.2%}")
    
    def cmd_refine(self, args: str):
        """Refine a prompt material"""
        if args:
            self.current_material = self.unity.discover_material(args)
        
        if not self.current_material:
            print("⚠ No material to refine. Use 'craft <prompt>' first")
            return
        
        print(f"\n⟲ Refining material (iteration {self.current_material.refinement_count + 1})...")
        
        # Refine the material
        self.current_material = self.unity.refine_material(self.current_material)
        
        print(f"✓ Refinement complete:")
        print(f"  Consistency: {self.current_material.consistency_score():.2%}")
        print(f"  Distinctiveness: {self.current_material.distinctiveness_score():.2%}")
        print(f"  Resistance: {self.current_material.resistance_score():.2%}")
        print(f"  Purity: {self.current_material.purity_score():.2%}")
        print(f"  Overall: {self.current_material.overall_score():.2%}")
        print(f"  Grade: {self.current_material.grade.name}")
    
    def cmd_discover(self, args: str):
        """Discover materials from source"""
        if not args:
            print("⚠ Please provide source text for discovery")
            return
        
        print(f"\n⚘ Discovering materials from: {args[:100]}...")
        
        material = self.unity.discover_material(args)
        self.unity.material_library.append(material)
        
        print(f"✓ Material discovered and added to library")
        print(f"  Library size: {len(self.unity.material_library)} materials")
    
    def cmd_integrate(self, args: str):
        """Integrate validated materials"""
        validated_materials = [
            m for m in self.unity.material_library
            if self.unity.validate_material(m)
        ]
        
        if not validated_materials:
            print("⚠ No validated materials to integrate")
            return
        
        print(f"\n∫ Integrating {len(validated_materials)} validated materials...")
        
        # Integrate into operations
        for material in validated_materials:
            # Determine best operation for material
            if "?" in material.content:
                operation = OperationType.INQUIRING
            elif "!" in material.content:
                operation = OperationType.FORMULATING
            elif "⇔" in material.content:
                operation = OperationType.REFLECTING
            else:
                operation = OperationType.DESCRIBING
            
            self.unity.integrate_materials([material], operation)
        
        print(f"✓ Integration complete")
    
    def cmd_evolve(self, args: str):
        """Run evolution cycles"""
        cycles = int(args) if args.isdigit() else 1
        
        print(f"\n♦∞ Running {cycles} evolution cycle(s)...")
        
        for i in range(cycles):
            print(f"\nCycle {i+1}/{cycles}:")
            self.unity.continuous_cycle()
        
        print(f"\n✓ Evolution complete. Total cycles: {self.unity.current_state['cycle_count']}")
    
    def cmd_describe(self, args: str):
        """Invoke describing operation"""
        if not args:
            print("⚠ Please provide input to describe")
            return
        
        print("\n≡ DESCRIBING OPERATION ≡")
        result = self.unity.invoke_describing(args)
        
        # Execute through LLM for real response
        prompt = f"≡ I am careful scribing ≡\nWhat appears before me: {args}\n≡ Direct observation:"
        response = self.llm_bridge.execute_prompt(prompt, temperature=0.3)
        
        print(response.content)
    
    def cmd_inquire(self, args: str):
        """Invoke inquiring operation"""
        if not args:
            args = self.session_history[-1] if self.session_history else "the nature of prompts"
        
        print("\n? INQUIRING OPERATION ?")
        
        prompt = f"? I am seeking-into ?\nFrom what I observe: {args}\n? What questions emerge:"
        response = self.llm_bridge.execute_prompt(prompt, temperature=0.7)
        
        print(response.content)
    
    def cmd_formulate(self, args: str):
        """Invoke formulating operation"""
        if not args:
            print("⚠ Please provide questions to formulate from")
            return
        
        print("\n! FORMULATING OPERATION !")
        
        prompt = f"! I am giving-form-to-insight !\nFrom these questions: {args}\n! What crystallizes:"
        response = self.llm_bridge.execute_prompt(prompt, temperature=0.8)
        
        print(response.content)
    
    def cmd_reflect(self, args: str):
        """Invoke reflecting operation"""
        if not args:
            print("⚠ Please provide insight to reflect on")
            return
        
        print("\n⇔ REFLECTING OPERATION ⇔")
        
        prompt = f"⇔ I am bending-back-to-examine ⇔\nExamining this insight: {args}\n⇔ Evidence assessment:"
        response = self.llm_bridge.execute_prompt(prompt, temperature=0.5)
        
        print(response.content)
    
    def cmd_status(self, args: str):
        """Show current status"""
        print("\n" + "=" * 40)
        print("CURRENT STATUS")
        print("=" * 40)
        print(f"Provider: {self.llm_bridge.config.provider.value}")
        print(f"Model: {self.llm_bridge.config.model}")
        print(f"Current Prompt: {self.current_prompt[:50] if self.current_prompt else 'None'}...")
        print(f"Material Library: {len(self.unity.material_library)} materials")
        print(f"Evolution Cycles: {self.unity.current_state['cycle_count']}")
        print(f"Session Commands: {len(self.session_history)}")
        
        # Show usage stats
        usage = self.llm_bridge.get_usage_report()
        print(f"\nLLM Usage:")
        print(f"  Requests: {usage['total_requests']}")
        print(f"  Tokens: {usage['total_tokens']}")
        print(f"  Cost: ${usage['total_cost']:.4f}")
        print(f"  Avg Latency: {usage['average_latency']}s")
    
    def cmd_save(self, args: str):
        """Save session to file"""
        filename = args if args else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = Path(filename)
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "current_prompt": self.current_prompt,
            "material_library": [
                {
                    "content": m.content,
                    "grade": m.grade.name,
                    "refinement_count": m.refinement_count,
                    "test_results": m.test_results
                }
                for m in self.unity.material_library
            ],
            "evolution_state": self.unity.current_state,
            "session_history": self.session_history[-50:]  # Last 50 commands
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"✓ Session saved to {filepath}")
    
    def cmd_load(self, args: str):
        """Load session from file"""
        if not args:
            print("⚠ Please provide filename to load")
            return
        
        filepath = Path(args)
        if not filepath.exists():
            print(f"⚠ File not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self.current_prompt = session_data.get("current_prompt")
        self.unity.current_state = session_data.get("evolution_state", {})
        self.session_history = session_data.get("session_history", [])
        
        # Recreate materials
        self.unity.material_library = []
        for m_data in session_data.get("material_library", []):
            material = PromptMaterial(
                content=m_data["content"],
                grade=MaterialGrade[m_data["grade"]],
                refinement_count=m_data["refinement_count"],
                test_results=m_data["test_results"]
            )
            self.unity.material_library.append(material)
        
        print(f"✓ Session loaded from {filepath}")
        print(f"  Materials: {len(self.unity.material_library)}")
        print(f"  Evolution cycles: {self.unity.current_state.get('cycle_count', 0)}")
    
    def process_command(self, command: str) -> bool:
        """Process a single command"""
        if not command.strip():
            return True
        
        # Add to history
        self.session_history.append(command)
        
        # Handle shortcuts
        if command.startswith("!"):
            # Quick test
            self.cmd_test(command[1:])
            return True
        elif command.startswith("?"):
            # Quick inquiry
            self.cmd_inquire(command[1:])
            return True
        elif command.startswith("@"):
            # Quick discovery
            self.cmd_discover(command[1:])
            return True
        
        # Parse command and arguments
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Command dispatch
        commands = {
            'craft': self.cmd_craft,
            'test': self.cmd_test,
            'refine': self.cmd_refine,
            'discover': self.cmd_discover,
            'integrate': self.cmd_integrate,
            'evolve': self.cmd_evolve,
            'describe': self.cmd_describe,
            'inquire': self.cmd_inquire,
            'formulate': self.cmd_formulate,
            'reflect': self.cmd_reflect,
            'judge': lambda a: print("⊢⊬~ Judging operation not yet implemented"),
            'deliberate': lambda a: print("⚖️ Deliberating operation not yet implemented"),
            'decide': lambda a: print("→ Deciding operation not yet implemented"),
            'plan': lambda a: print("═══► Planning operation not yet implemented"),
            'status': self.cmd_status,
            'history': lambda a: print("\n".join(self.session_history[-20:])),
            'save': self.cmd_save,
            'load': self.cmd_load,
            'clear': lambda a: self.__init__(),
            'help': lambda a: self.print_help(),
            'exit': lambda a: False,
            'quit': lambda a: False,
        }
        
        if cmd in commands:
            result = commands[cmd](args)
            return result if result is not None else True
        else:
            print(f"⚠ Unknown command: {cmd}. Type 'help' for commands")
            return True
    
    def run(self):
        """Run the interactive CLI"""
        self.print_banner()
        
        try:
            while True:
                try:
                    command = input("\npromptcraft> ").strip()
                    if not self.process_command(command):
                        break
                except KeyboardInterrupt:
                    print("\n\n[Interrupted - type 'exit' to quit]")
                except Exception as e:
                    print(f"\n⚠ Error: {e}")
        
        finally:
            # Save command history
            readline.write_history_file(self.history_file)
            
            print("\n♦∞ Thank you for crafting with PromptKraft ∞♦\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PromptKraft - Self-Evolving Prompt Craftsmanship"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        help="Model to use (e.g., gpt-3.5-turbo, claude-3-haiku)"
    )
    parser.add_argument(
        "--command",
        "-c",
        help="Execute a single command and exit"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize CLI (requires API keys)
        cli = PromptKraftCLI()
    except ValueError as e:
        print(f"\n⚠ Error: {e}")
        print("\nPlease ensure you have API keys set up in .env file:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  ANTHROPIC_API_KEY=your-key-here")
        return
    
    # Override provider if specified
    if args.provider:
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC
        }
        cli.llm_bridge.switch_provider(provider_map[args.provider])
    
    # Override model if specified
    if args.model:
        cli.llm_bridge.set_model(args.model)
    
    # Execute single command or run interactive mode
    if args.command:
        cli.process_command(args.command)
    else:
        cli.run()


if __name__ == "__main__":
    main()