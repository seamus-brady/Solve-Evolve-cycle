"""Tests for agent/solver.py"""

from pathlib import Path
from typing import Any, Dict

import pytest

from agent.observations import ObservationBuffer
from agent.registry import KnowledgeRegistry, ToolRegistry
from agent.solver import Solver, TaskResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

ADD_TOOL_CODE = """\
def add(a: int = 0, b: int = 0) -> int:
    return a + b
"""

GREET_TOOL_CODE = """\
def greet(name: str = "World") -> str:
    return f"Hello, {name}!"
"""

FAILING_TOOL_CODE = """\
def bad_tool(x: int = 0) -> int:
    raise ValueError(f"bad_tool always fails with x={x}")
"""

PRINT_TOOL_CODE = """\
def print_tool(msg: str = "") -> str:
    print(f"printed: {msg}")
    return "done"
"""


@pytest.fixture
def tmp_dirs(tmp_path):
    return {
        "tools": tmp_path / "tools",
        "knowledge": tmp_path / "knowledge.json",
        "logs": tmp_path / "obs.jsonl",
    }


@pytest.fixture
def solver(tmp_dirs) -> Solver:
    tool_reg = ToolRegistry(tmp_dirs["tools"])
    knowledge_reg = KnowledgeRegistry(tmp_dirs["knowledge"])
    buf = ObservationBuffer(tmp_dirs["logs"])
    return Solver(tool_reg, knowledge_reg, buf)


@pytest.fixture
def solver_with_tools(tmp_dirs) -> Solver:
    tool_reg = ToolRegistry(tmp_dirs["tools"])
    tool_reg.create_tool("add", ADD_TOOL_CODE)
    tool_reg.create_tool("greet", GREET_TOOL_CODE)

    knowledge_reg = KnowledgeRegistry(tmp_dirs["knowledge"])
    buf = ObservationBuffer(tmp_dirs["logs"])
    return Solver(tool_reg, knowledge_reg, buf)


# ── TaskResult structure ──────────────────────────────────────────────────────


class TestTaskResultStructure:
    def test_successful_result_fields(self, solver_with_tools):
        result = solver_with_tools.execute("add", tool_name="add", args={"a": 3, "b": 4})
        assert result.task == "add"
        assert result.outcome == "success"
        assert result.result == 7
        assert result.error is None
        assert isinstance(result.episode_id, str)
        assert len(result.episode_id) > 0

    def test_failed_result_fields(self, tmp_dirs):
        tool_reg = ToolRegistry(tmp_dirs["tools"])
        tool_reg.create_tool("bad_tool", FAILING_TOOL_CODE)
        buf = ObservationBuffer(tmp_dirs["logs"])
        s = Solver(tool_reg, KnowledgeRegistry(tmp_dirs["knowledge"]), buf)

        result = s.execute("run bad tool", tool_name="bad_tool", args={"x": 5})
        assert result.outcome == "failure"
        assert result.error is not None
        assert "bad_tool" in result.error or "5" in result.error
        assert result.result is None

    def test_episode_id_unique_per_call(self, solver_with_tools):
        r1 = solver_with_tools.execute("add", tool_name="add", args={"a": 1, "b": 2})
        r2 = solver_with_tools.execute("add", tool_name="add", args={"a": 3, "b": 4})
        assert r1.episode_id != r2.episode_id


# ── Observation logging ───────────────────────────────────────────────────────


class TestObservationLogging:
    def test_execute_logs_observation(self, solver_with_tools, tmp_dirs):
        buf = ObservationBuffer(tmp_dirs["logs"])
        solver_with_tools.execute("add", tool_name="add", args={"a": 1, "b": 2})
        obs_list = buf.read_all()
        assert len(obs_list) == 1
        assert obs_list[0].task == "add"
        assert obs_list[0].outcome == "success"

    def test_failure_logged_as_failure(self, tmp_dirs):
        tool_reg = ToolRegistry(tmp_dirs["tools"])
        tool_reg.create_tool("bad_tool", FAILING_TOOL_CODE)
        buf = ObservationBuffer(tmp_dirs["logs"])
        s = Solver(tool_reg, KnowledgeRegistry(tmp_dirs["knowledge"]), buf)

        s.execute("run bad", tool_name="bad_tool", args={"x": 0})
        obs_list = buf.read_all()
        assert obs_list[0].outcome == "failure"

    def test_multiple_executions_all_logged(self, solver_with_tools, tmp_dirs):
        buf = ObservationBuffer(tmp_dirs["logs"])
        for i in range(3):
            solver_with_tools.execute("add", tool_name="add", args={"a": i, "b": 1})
        assert buf.count() == 3

    def test_observation_contains_tool_calls(self, solver_with_tools, tmp_dirs):
        buf = ObservationBuffer(tmp_dirs["logs"])
        solver_with_tools.execute("add numbers", tool_name="add", args={"a": 5, "b": 6})
        obs = buf.read_all()[0]
        assert len(obs.tool_calls) == 1
        assert obs.tool_calls[0].tool_name == "add"
        assert obs.tool_calls[0].input == {"a": 5, "b": 6}

    def test_tool_call_output_recorded(self, solver_with_tools, tmp_dirs):
        buf = ObservationBuffer(tmp_dirs["logs"])
        solver_with_tools.execute("greet", tool_name="greet", args={"name": "Alice"})
        obs = buf.read_all()[0]
        assert "Hello, Alice!" in obs.tool_calls[0].output

    def test_error_logged_in_tool_call(self, tmp_dirs):
        tool_reg = ToolRegistry(tmp_dirs["tools"])
        tool_reg.create_tool("bad_tool", FAILING_TOOL_CODE)
        buf = ObservationBuffer(tmp_dirs["logs"])
        s = Solver(tool_reg, KnowledgeRegistry(tmp_dirs["knowledge"]), buf)

        s.execute("bad", tool_name="bad_tool", args={"x": 1})
        obs = buf.read_all()[0]
        assert obs.tool_calls[0].error is not None


# ── Tool dispatch ─────────────────────────────────────────────────────────────


class TestToolDispatch:
    def test_explicit_tool_name(self, solver_with_tools):
        result = solver_with_tools.execute("compute", tool_name="add", args={"a": 10, "b": 20})
        assert result.result == 30

    def test_implicit_dispatch_by_task_substring(self, solver_with_tools):
        # "add" is a substring of "please add two numbers"
        result = solver_with_tools.execute("please add two numbers", args={"a": 2, "b": 3})
        assert result.result == 5

    def test_implicit_dispatch_by_tool_name_match(self, solver_with_tools):
        result = solver_with_tools.execute("greet someone", args={"name": "Bob"})
        assert result.result == "Hello, Bob!"

    def test_unknown_tool_returns_failure(self, solver_with_tools):
        result = solver_with_tools.execute("use nonexistent_tool", tool_name="nonexistent")
        assert result.outcome == "failure"
        assert result.error is not None

    def test_no_matching_tool_in_task_returns_failure(self, solver_with_tools):
        result = solver_with_tools.execute("do something unrelated")
        assert result.outcome == "failure"

    def test_no_tools_registered_returns_failure(self, solver):
        result = solver.execute("add", tool_name="add", args={"a": 1, "b": 2})
        assert result.outcome == "failure"


# ── stdout/stderr capture ─────────────────────────────────────────────────────


class TestOutputCapture:
    def test_stdout_captured_in_observation(self, tmp_dirs):
        tool_reg = ToolRegistry(tmp_dirs["tools"])
        tool_reg.create_tool("print_tool", PRINT_TOOL_CODE)
        buf = ObservationBuffer(tmp_dirs["logs"])
        s = Solver(tool_reg, KnowledgeRegistry(tmp_dirs["knowledge"]), buf)

        result = s.execute("print", tool_name="print_tool", args={"msg": "hi"})
        assert result.outcome == "success"
        obs = buf.read_all()[0]
        assert "printed: hi" in obs.tool_calls[0].output

    def test_stdout_not_leaked_to_real_stdout(self, tmp_dirs, capsys):
        tool_reg = ToolRegistry(tmp_dirs["tools"])
        tool_reg.create_tool("print_tool", PRINT_TOOL_CODE)
        buf = ObservationBuffer(tmp_dirs["logs"])
        s = Solver(tool_reg, KnowledgeRegistry(tmp_dirs["knowledge"]), buf)

        s.execute("print", tool_name="print_tool", args={"msg": "secret"})
        captured = capsys.readouterr()
        assert "secret" not in captured.out


# ── Args handling ─────────────────────────────────────────────────────────────


class TestArgsHandling:
    def test_empty_args_uses_defaults(self, solver_with_tools):
        result = solver_with_tools.execute("greet", tool_name="greet", args={})
        assert result.result == "Hello, World!"

    def test_none_args_uses_defaults(self, solver_with_tools):
        result = solver_with_tools.execute("greet", tool_name="greet", args=None)
        assert result.result == "Hello, World!"
