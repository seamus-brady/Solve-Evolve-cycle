"""
Evolver — the Evolve phase of the Solve-Evolve cycle.

Triggered asynchronously after W solve episodes. Runs a four-step LLM pipeline:

    1. Diagnose  — identify persistent failure patterns in the last W observations
    2. Plan      — map each pattern to a typed edit operation (CreateTool, etc.)
    3. Update    — apply the planned edits to the registries
    4. Verify    — run the validation suite; only commit if all tests pass

The evolver has a separate compute budget from the solver and never interferes
with ongoing task execution.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore

from .observations import Observation, ObservationBuffer
from .registry import KnowledgeRegistry, ValidationResult, ToolRegistry, ValidationRegistry

# Default window size W from the paper
DEFAULT_WINDOW_SIZE = 10
DEFAULT_MODEL = "claude-opus-4-6"


@dataclass
class EvolveResult:
    """Summary of one evolver run."""

    diagnosed: bool
    plans: List[Dict[str, Any]]
    updates_applied: int
    tests_passed: bool
    committed: bool
    output: str
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)


class Evolver:
    """
    Asynchronous artifact improvement pipeline.

    Inject a pre-built ``anthropic.Anthropic`` client (or any object whose
    ``messages.create()`` accepts the same kwargs) so the evolver can be
    tested without real API calls.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        knowledge_registry: KnowledgeRegistry,
        validation_registry: ValidationRegistry,
        observation_buffer: ObservationBuffer,
        client: Any = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.tool_registry = tool_registry
        self.knowledge_registry = knowledge_registry
        self.validation_registry = validation_registry
        self.observation_buffer = observation_buffer
        self.window_size = window_size
        self.model = model

        if client is not None:
            self.client = client
        elif anthropic is not None:
            self.client = anthropic.Anthropic()
        else:
            raise RuntimeError(
                "anthropic package not installed and no client was provided."
            )

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> EvolveResult:
        """
        Run the full four-step evolver pipeline.

        Returns an EvolveResult describing what was diagnosed, planned,
        applied, and whether the changes were committed.
        """
        observations = self.observation_buffer.read_last_n(self.window_size)

        if not observations:
            return EvolveResult(
                diagnosed=False,
                plans=[],
                updates_applied=0,
                tests_passed=False,
                committed=False,
                output="No observations in buffer — nothing to evolve.",
            )

        # ── Step 1: Diagnose ──────────────────────────────────────────────
        failure_patterns = self._diagnose(observations)

        if not failure_patterns:
            return EvolveResult(
                diagnosed=True,
                plans=[],
                updates_applied=0,
                tests_passed=True,
                committed=False,
                output="No persistent failure patterns detected.",
                failure_patterns=[],
            )

        # ── Step 2: Plan ──────────────────────────────────────────────────
        edit_plans = self._plan(failure_patterns)

        if not edit_plans:
            return EvolveResult(
                diagnosed=True,
                plans=[],
                updates_applied=0,
                tests_passed=True,
                committed=False,
                output="Failure patterns found but no edits planned.",
                failure_patterns=failure_patterns,
            )

        # ── Step 3: Update ────────────────────────────────────────────────
        updates_applied = self._update(edit_plans)

        # ── Step 4: Verify ────────────────────────────────────────────────
        test_result = self.validation_registry.run_tests()

        return EvolveResult(
            diagnosed=True,
            plans=edit_plans,
            updates_applied=updates_applied,
            tests_passed=test_result.passed,
            committed=test_result.passed,
            output=test_result.output,
            failure_patterns=failure_patterns,
        )

    # ── Step implementations ──────────────────────────────────────────────

    def _diagnose(self, observations: List[Observation]) -> List[Dict[str, Any]]:
        """
        Ask the LLM to identify *persistent* failure patterns.

        Batches the last W observation windows and returns a list of pattern
        dicts, each with keys: pattern, frequency, affected_tools, severity.
        """
        obs_summary = [
            {
                "task": obs.task,
                "outcome": obs.outcome,
                "error": obs.error,
                "tool_calls": [
                    {"name": tc.tool_name, "error": tc.error}
                    for tc in obs.tool_calls
                ],
            }
            for obs in observations
        ]

        prompt = (
            f"Analyze these {len(observations)} agent observations and identify "
            "PERSISTENT failure patterns (not one-off noise).\n\n"
            f"Observations:\n{json.dumps(obs_summary, indent=2)}\n\n"
            "Return a JSON array of failure patterns. Each object must have:\n"
            '- "pattern": string description of the failure\n'
            '- "frequency": integer count of occurrences\n'
            '- "affected_tools": list of tool names involved\n'
            '- "severity": "high" | "medium" | "low"\n\n'
            "Return ONLY the JSON array. Return [] if no persistent failures found."
        )

        return self._call_llm_for_json(prompt, expected_type=list)

    def _plan(
        self, failure_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map each failure pattern to a typed edit operation.

        Returns a dependency-ordered list of edit operation dicts.
        """
        existing_tools = self.tool_registry.list_tools()
        existing_knowledge = self.knowledge_registry.list_keys()

        prompt = (
            "Given these persistent failure patterns, plan the minimal set of "
            "artifact edits needed to fix them.\n\n"
            f"Failure Patterns:\n{json.dumps(failure_patterns, indent=2)}\n\n"
            f"Existing Tools: {existing_tools}\n"
            f"Existing Knowledge Keys: {existing_knowledge}\n\n"
            "Return a dependency-ordered JSON array of edit operations.\n"
            "Each operation must have:\n"
            '- "op_type": one of "CreateTool" | "EvolveTool" | "AddKnowledge" '
            '| "EvolveKnowledge" | "PruneTool"\n'
            "- CreateTool: name, code (Python function string), description\n"
            "- EvolveTool: name (existing tool), new_code, reason\n"
            "- AddKnowledge: key, value, description\n"
            "- EvolveKnowledge: key, new_value, reason\n"
            "- PruneTool: name, reason\n\n"
            "Return ONLY the JSON array."
        )

        return self._call_llm_for_json(prompt, expected_type=list)

    def _update(self, edit_plans: List[Dict[str, Any]]) -> int:
        """
        Apply planned edits to the registries.

        Returns the number of edits successfully applied.
        """
        applied = 0

        for plan in edit_plans:
            op_type = plan.get("op_type")
            try:
                if op_type == "CreateTool":
                    self.tool_registry.create_tool(
                        name=plan["name"],
                        code=plan["code"],
                        description=plan.get("description", ""),
                    )
                    applied += 1

                elif op_type == "EvolveTool":
                    self.tool_registry.create_tool(
                        name=plan["name"],
                        code=plan["new_code"],
                    )
                    applied += 1

                elif op_type == "AddKnowledge":
                    self.knowledge_registry.set(
                        key=plan["key"],
                        value=plan["value"],
                        description=plan.get("description", ""),
                        provenance="evolver",
                    )
                    applied += 1

                elif op_type == "EvolveKnowledge":
                    self.knowledge_registry.set(
                        key=plan["key"],
                        value=plan["new_value"],
                        provenance="evolver",
                    )
                    applied += 1

                elif op_type == "PruneTool":
                    self.tool_registry.delete_tool(plan["name"])
                    applied += 1

            except (KeyError, TypeError, Exception):
                # Skip malformed edit ops and continue
                continue

        return applied

    # ── helpers ───────────────────────────────────────────────────────────

    def _call_llm_for_json(
        self, prompt: str, expected_type: type = list
    ) -> Any:
        """
        Send a prompt to the LLM and parse the response as JSON.

        Uses adaptive thinking on Opus 4.6 and streaming for robustness.
        Falls back to an empty instance of expected_type on parse failure.
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = stream.get_final_message()

        # Extract text from the response (skip thinking blocks)
        text = next(
            (
                block.text
                for block in response.content
                if hasattr(block, "text") and block.type == "text"
            ),
            "",
        )

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, expected_type):
                return expected_type()
            return parsed
        except (json.JSONDecodeError, ValueError):
            return expected_type()
