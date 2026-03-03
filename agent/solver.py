"""
Solver — the Solve phase of the Solve-Evolve cycle.

Loads the registries read-only, executes tasks by dispatching to registered
tools, captures all output, and appends a structured Observation to the log.

Compute budget note: the Solver never calls an LLM directly. It only executes
deterministic tool functions. LLM calls (if any) belong in the tools themselves.
"""

import io
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .observations import Observation, ObservationBuffer, ToolCall
from .registry import KnowledgeRegistry, ToolRegistry


@dataclass
class TaskResult:
    """The outcome of a single task execution."""

    task: str
    outcome: str  # "success" | "failure"
    result: Any
    error: Optional[str]
    tool_calls: List[ToolCall]
    episode_id: str


class Solver:
    """
    Execute tasks read-only against the current registries.

    The solver is intentionally stateless between calls — all state lives
    in the registries and the observation buffer.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        knowledge_registry: KnowledgeRegistry,
        observation_buffer: ObservationBuffer,
    ) -> None:
        self.tool_registry = tool_registry
        self.knowledge_registry = knowledge_registry
        self.observation_buffer = observation_buffer

    def execute(
        self,
        task: str,
        tool_name: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Execute a task, log the observation, and return the result.

        Parameters
        ----------
        task:
            Human-readable task description (logged verbatim).
        tool_name:
            Name of the tool to use. If omitted, the solver tries to match
            a tool by name substring against ``task``.
        args:
            Keyword arguments forwarded to the tool function.
        """
        episode_id = str(uuid.uuid4())
        tool_calls: List[ToolCall] = []
        args = args or {}

        outcome = "failure"
        error: Optional[str] = None
        result_value: Any = None

        try:
            result_value = self._dispatch(task, tool_name, args, tool_calls)
            outcome = "success"
        except Exception as exc:
            error = str(exc)

        obs = Observation(
            task=task,
            tool_calls=tool_calls,
            outcome=outcome,
            error=error,
            timestamp=time.time(),
            episode_id=episode_id,
        )
        self.observation_buffer.append(obs)

        return TaskResult(
            task=task,
            outcome=outcome,
            result=result_value,
            error=error,
            tool_calls=tool_calls,
            episode_id=episode_id,
        )

    # ── internal ─────────────────────────────────────────────────────────

    def _dispatch(
        self,
        task: str,
        tool_name: Optional[str],
        args: Dict[str, Any],
        tool_calls: List[ToolCall],
    ) -> Any:
        """Resolve which tool to call and invoke it."""
        tools = self.tool_registry.load_all_tools()

        if tool_name is not None:
            # Explicit tool requested
            func = tools.get(tool_name)
            if func is None:
                raise ValueError(f"Tool '{tool_name}' not found in registry.")
            return self._call_tool(tool_name, func, args, tool_calls)

        # Implicit dispatch: substring match on task description
        for name, func in tools.items():
            if name.lower() in task.lower():
                return self._call_tool(name, func, args, tool_calls)

        raise ValueError(
            f"No suitable tool found for task: '{task}'. "
            f"Available tools: {list(tools.keys())}"
        )

    def _call_tool(
        self,
        name: str,
        func: Callable,
        args: Dict[str, Any],
        tool_calls: List[ToolCall],
    ) -> Any:
        """
        Invoke a tool, capturing stdout/stderr, and record the ToolCall.

        Raises RuntimeError if the tool raises.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        result_value: Any = None
        tool_error: Optional[str] = None

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        try:
            result_value = func(**args)
        except Exception as exc:
            tool_error = str(exc)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()

        output_parts = []
        if result_value is not None:
            output_parts.append(str(result_value))
        if captured_stdout:
            output_parts.append(f"stdout: {captured_stdout}")
        if captured_stderr:
            output_parts.append(f"stderr: {captured_stderr}")
        full_output = "\n".join(output_parts)

        tool_calls.append(
            ToolCall(
                tool_name=name,
                input=args,
                output=full_output,
                error=tool_error,
            )
        )

        if tool_error:
            raise RuntimeError(f"Tool '{name}' raised: {tool_error}")

        return result_value
