# Solve-Evolve Cycle

A Python implementation of the **A-Evolve** agentic evolution framework, based on the paper ["Agentic Evolution is the Path to Evolving LLMs"](https://arxiv.org/abs/2602.00359v1).

The core idea: instead of rebuilding solutions from scratch on every run, an agent accumulates verified tools and knowledge across episodes, then asynchronously reflects on failures to improve its own artifacts.

## How It Works

The system alternates between two phases with separate compute budgets:

**Solve** — The agent executes tasks by dispatching to registered tools. All inputs, outputs, errors, and outcomes are logged to an append-only observation buffer. The solver never calls an LLM directly; it only runs deterministic tool functions.

**Evolve** — After `W` episodes (default: 10), a separate evolver process reads the observation buffer, identifies persistent failure patterns, and proposes artifact updates through a four-step pipeline:

1. **Diagnose** — LLM analyzes the last `W` observations to find recurring (not one-off) failures
2. **Plan** — Maps each failure to a typed edit operation (`CreateTool`, `EvolveTool`, `AddKnowledge`, `EvolveKnowledge`, `PruneTool`)
3. **Update** — Applies the planned edits to the registries
4. **Verify** — Runs the validation suite with pytest; only commits if all tests pass

## Architecture

```
agent/
  registries/
    knowledge.json      # versioned facts, schemas, worked examples
    tools/              # Python files — one tool per file
    tests/              # pytest files — one test file per tool
  logs/
    observations.jsonl  # append-only structured trace log
  solver.py             # Solve phase: loads registries, executes tasks, logs observations
  evolver.py            # Evolve phase: diagnose → plan → update → verify
  registry.py           # KnowledgeRegistry, ToolRegistry, ValidationRegistry
  observations.py       # Observation + ObservationBuffer
  edit_ops.py           # Typed edit operation dataclasses
```

### Three Persistent Registries

| Registry | Contents |
|---|---|
| **Knowledge** (`K`) | Schemas, workflows, interface contracts, worked examples |
| **Tool** (`T`) | Executable Python functions — dynamically loaded, hot-reloadable |
| **Validation** (`V`) | pytest files, one per tool — run before any artifact is committed |

All registries live on disk, are version-tracked, and are **read-only during Solve** but writable during Evolve.

### Typed Edit Operations

The evolver produces structured edit plans rather than open-ended self-modification:

| Operation | Effect |
|---|---|
| `CreateTool` | Add a new tool function to the registry |
| `EvolveTool` | Replace the code of an existing tool |
| `AddKnowledge` | Add a new knowledge entry |
| `EvolveKnowledge` | Update an existing knowledge entry |
| `PruneTool` | Remove a tool that is no longer useful |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and an `ANTHROPIC_API_KEY` environment variable for the Evolver.

## Usage

```python
from pathlib import Path
from agent.registry import KnowledgeRegistry, ToolRegistry, ValidationRegistry
from agent.observations import ObservationBuffer
from agent.solver import Solver
from agent.evolver import Evolver

# Set up registries
tool_reg = ToolRegistry(Path("agent/registries/tools"))
know_reg = KnowledgeRegistry(Path("agent/registries/knowledge.json"))
val_reg = ValidationRegistry(Path("agent/registries/tests"))
obs_buf = ObservationBuffer(Path("agent/logs/observations.jsonl"))

# Solve phase: run a task
solver = Solver(tool_reg, know_reg, obs_buf)
result = solver.execute("my_tool", tool_name="my_tool", args={"x": 42})

# Evolve phase: improve artifacts after W episodes
evolver = Evolver(tool_reg, know_reg, val_reg, obs_buf)
evolve_result = evolver.run()
print(evolve_result.committed)  # True if tests passed and changes were committed
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Design Principles

- **Separate solve and evolve compute budgets** — reflection never interrupts task execution
- **Batch failures before acting** — `W=10` distinguishes real capability gaps from noise
- **Type your edit operations** — structured ops give the LLM cleaner output targets than open-ended prompts
- **Never commit without tests passing** — validation gating prevents uncontrolled drift
- **Log structured traces** — tool calls, stdout/stderr, and error signals are all captured for diagnosis

## Reference

> "Agentic Evolution is the Path to Evolving LLMs" — arXiv:2602.00359v1
