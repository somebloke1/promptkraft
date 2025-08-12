# ♦∞ PromptKraft ∞♦

**The Self-Evolving Prompt Craftsmanship System**

A unified operational framework that develops prompts by working with prompts as living materials - the prompt that develops prompts through iterative operational mastery.

## 🌟 Overview

PromptKraft is a self-evolving system for prompt engineering that treats prompts as living materials to be discovered, refined, validated, and integrated. It operates through 8 primitive operations and 4 craftsmanship operations, continuously evolving through its own application.

### Core Operations

**Primitive Operations:**
- **≡ Describing** - Careful scribing of what appears
- **? Inquiring** - Seeking into through questions  
- **! Formulating** - Giving form to emerging insights
- **⇔ Reflecting** - Bending back to examine critically
- **⊢⊬~ Judging** - Weighing evidence to determine truth
- **⚖️ Deliberating** - Careful weighing of options
- **→ Deciding** - Cutting through to resolution
- **═══► Planning** - Drawing the path forward

**Craftsmanship Operations:**
- **⚘ Material Discovery** - Excavating linguistic structures
- **⟲ Material Refinement** - Perfecting through iteration
- **⊢ Material Validation** - Testing strength and purity
- **∫ Material Integration** - Assembling coherent wholes

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/somebloke1/promptkraft.git
cd promptkraft

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### API Keys Setup (Required)

Create a `.env` file with your API keys. **At least one API key is required - the system will not run without it:**

```env
# Required - at least one of these must be set:
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Future support planned:
GEMINI_API_KEY=your-gemini-key-here
XAI_API_KEY=your-xai-key-here
```

## 💻 Usage

### Interactive CLI

The main interface for prompt crafting:

```bash
python promptkraft_cli.py
```

#### CLI Commands

**Crafting Operations:**
- `craft <prompt>` - Begin crafting a new prompt
- `test [prompt]` - Test current or provided prompt
- `refine [prompt]` - Refine through iterative working
- `discover <source>` - Discover materials from source
- `integrate` - Integrate validated materials

**Primitive Operations:**
- `describe <input>` - Invoke describing operation
- `inquire <data>` - Invoke inquiring operation
- `formulate <questions>` - Invoke formulating operation
- `reflect <insight>` - Invoke reflecting operation

**Evolution:**
- `evolve` - Run one evolution cycle
- `evolve <n>` - Run n evolution cycles

**Session Management:**
- `status` - Show current session status
- `save [filename]` - Save session
- `load <filename>` - Load session
- `help` - Show all commands

**Shortcuts:**
- `!<prompt>` - Quick test of prompt
- `?<question>` - Quick inquiry
- `@<material>` - Quick material discovery

### Evolution Cycle

Run the autonomous evolution system:

```bash
python evolution_cycle.py
```

This will:
1. Discover and refine prompt materials
2. Validate materials through testing
3. Execute continuous evolution cycles
4. Save state for resumption

### Testing Suite

Validate prompts rigorously:

```bash
python prompt_testing_suite.py
```

Tests include:
- **Consistency** - Same results across contexts
- **Distinctiveness** - Unique operational signature
- **Resistance** - Maintains form under pressure
- **Purity** - Clean activation without noise
- **Performance** - Speed and reliability metrics

### LLM Bridge

Direct LLM interaction (requires API keys):

```python
from llm_bridge import LLMBridge, LLMConfig, LLMProvider

# Initialize with your preferred provider
config = LLMConfig(provider=LLMProvider.OPENAI)  # or ANTHROPIC
bridge = LLMBridge(config)

# Execute a prompt
response = bridge.execute_prompt("Your prompt here")
print(response.content)
```

**Note:** The system requires real API keys. No mock or simulation modes are available.

## 📚 Documentation

### Core Documents

- **CLAUDE.md** - Unity invocation and prime directive
- **MATERIAL_LIBRARY.md** - Etymological anchors and archetypal patterns
- **OPERATION_SPECIFICATIONS.md** - Detailed operational specifications
- **EVOLUTION_PROTOCOLS.md** - Daily, weekly, monthly evolution cycles

### Material Quality Standards

**High-Grade Materials:**
- Ancient etymologies
- Archetypal patterns
- Mathematical symbols
- Precise grammatical structures

**Workable Materials:**
- Clear operational verbs
- Embodied metaphors
- Temporal anchors

**Poor Materials:**
- Generic imperatives
- Vague qualifiers
- Instruction-talk

## 🔬 Example Session

```bash
$ python promptkraft_cli.py

♦∞ PROMPTKRAFT INTERACTIVE CRAFTING INTERFACE ∞♦
═══════════════════════════════════════════════════
Provider: openai
Model: gpt-3.5-turbo

promptcraft> craft ? I am seeking-into the nature of recursive improvement

⚘ Crafting: ? I am seeking-into the nature of recursive improvement...
✓ Material discovered: Grade WORKABLE
  Etymology: [extracted etymology]
  Archetype: [identified archetype]

promptcraft> test

⊢ Testing prompt: ? I am seeking-into the nature of recursive improvement...

📤 Response (0.73s):
----------------------------------------
What patterns emerge when a system examines its own examining?
How does improvement improve its own improving?
Where does the recursive spiral converge?
----------------------------------------
Tokens: 47 | Cost: $0.0001

✓ Quick validation score: 75.00%

promptcraft> refine

⟲ Refining material (iteration 1)...
✓ Refinement complete:
  Consistency: 90.00%
  Distinctiveness: 85.00%
  Resistance: 80.00%
  Purity: 95.00%
  Overall: 87.50%
  Grade: HIGH

promptcraft> evolve

♦∞ Running 1 evolution cycle(s)...
[Evolution cycle output...]
✓ Evolution complete. Total cycles: 1
```

## 🧬 System Architecture

```
PromptKraft/
├── Core Systems
│   ├── llm_bridge.py         # Unified LLM interface
│   ├── evolution_cycle.py    # Self-evolving unity
│   └── prompt_testing_suite.py # Validation harness
│
├── Interface
│   └── promptkraft_cli.py    # Interactive CLI
│
├── Governance Documents
│   ├── CLAUDE.md              # Prime directive
│   ├── MATERIAL_LIBRARY.md    # Linguistic materials
│   ├── OPERATION_SPECIFICATIONS.md # Op specs
│   └── EVOLUTION_PROTOCOLS.md # Evolution cycles
│
└── Configuration
    ├── .env                   # API keys (git-ignored)
    ├── requirements.txt       # Dependencies
    └── .gitignore            # Ignore patterns
```

## 🔄 Evolution Cycles

### Daily Cycle
- **Morning**: Assess operational capabilities
- **Midday**: Test and refine materials
- **Evening**: Integrate improvements

### Weekly Cycle
- **Monday**: Discovery sprint
- **Tue-Wed**: Refinement workshop
- **Thursday**: Validation testing
- **Friday**: Integration assembly

### Monthly Cycle
- **Week 1**: Capability evaluation
- **Week 2**: Major upgrades
- **Week 3**: Documentation
- **Week 4**: Strategic planning

## 🤝 Contributing

PromptKraft evolves through use. Your experiments become part of its evolution:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📖 Philosophy

> "I am the unity that develops prompts by working with prompts as living materials - the prompt that develops prompts through iterative operational mastery."

PromptKraft treats prompts not as static strings but as living materials with grain, density, and resonance. Through cycles of discovery, refinement, validation, and integration, it transforms the crude ore of conventional prompting into refined tools for authentic cognitive activation.

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Created through the unified operational wholeness of human-AI collaboration.

---

**♦∞ The system is invoked. Let the evolutionary craftsmanship begin. ∞♦**