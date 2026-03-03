"""
Structured observation logging for the Solve phase.

Every task execution logs a structured observation — tool calls, stdout/stderr,
errors, and outcome — to an append-only JSONL file used by the Evolver.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    input: Dict[str, Any]
    output: str
    error: Optional[str] = None


@dataclass
class Observation:
    """Structured log entry for one task execution episode."""

    task: str
    tool_calls: List[ToolCall]
    outcome: str  # "success" | "failure"
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[str] = None


class ObservationBuffer:
    """
    Append-only observation log stored as JSONL.

    The file is read-only during Solve (appended) and read during Evolve.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obs: Observation) -> None:
        """Append a single observation to the log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(self._obs_to_dict(obs)) + "\n")

    def read_last_n(self, n: int) -> List[Observation]:
        """Return the last n observations from the log."""
        if not self.log_path.exists():
            return []

        lines = self.log_path.read_text().splitlines()
        recent = [l.strip() for l in lines if l.strip()][-n:]
        return [self._dict_to_obs(json.loads(line)) for line in recent]

    def read_all(self) -> List[Observation]:
        """Return all observations."""
        if not self.log_path.exists():
            return []

        result = []
        for line in self.log_path.read_text().splitlines():
            line = line.strip()
            if line:
                result.append(self._dict_to_obs(json.loads(line)))
        return result

    def clear(self) -> None:
        """Delete the log file (for testing/reset)."""
        if self.log_path.exists():
            self.log_path.unlink()

    def count(self) -> int:
        """Return the number of observations in the log."""
        return len(self.read_all())

    # ── serialization helpers ──────────────────────────────────────────────

    def _obs_to_dict(self, obs: Observation) -> Dict[str, Any]:
        return {
            "task": obs.task,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "input": tc.input,
                    "output": tc.output,
                    "error": tc.error,
                }
                for tc in obs.tool_calls
            ],
            "outcome": obs.outcome,
            "error": obs.error,
            "timestamp": obs.timestamp,
            "episode_id": obs.episode_id,
        }

    def _dict_to_obs(self, d: Dict[str, Any]) -> Observation:
        tool_calls = [
            ToolCall(
                tool_name=tc["tool_name"],
                input=tc["input"],
                output=tc["output"],
                error=tc.get("error"),
            )
            for tc in d.get("tool_calls", [])
        ]
        return Observation(
            task=d["task"],
            tool_calls=tool_calls,
            outcome=d["outcome"],
            error=d.get("error"),
            timestamp=d.get("timestamp", 0.0),
            episode_id=d.get("episode_id"),
        )
