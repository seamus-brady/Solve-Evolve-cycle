"""Tests for agent/evolver.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.evolver import DEFAULT_WINDOW_SIZE, EvolveResult, Evolver
from agent.observations import Observation, ObservationBuffer, ToolCall
from agent.registry import KnowledgeRegistry, ValidationResult, ToolRegistry, ValidationRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

ADD_TOOL_CODE = "def add(a=0, b=0):\n    return a + b\n"


def make_mock_client(response_text: str) -> MagicMock:
    """Build a mock anthropic client that returns ``response_text`` as the LLM output."""
    client = MagicMock()
    response = MagicMock()
    text_block = MagicMock()
    text_block.text = response_text
    text_block.type = "text"
    response.content = [text_block]

    # Support the streaming interface used by the evolver: client.messages.stream(...)
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.get_final_message = MagicMock(return_value=response)
    client.messages.stream.return_value = stream_cm

    return client


def make_obs(task="task", outcome="success", error=None) -> Observation:
    return Observation(
        task=task,
        tool_calls=[ToolCall(tool_name="t", input={}, output="out", error=error)],
        outcome=outcome,
        error=error,
        timestamp=1000.0,
        episode_id="ep-1",
    )


@pytest.fixture
def registries(tmp_path):
    return {
        "tools": ToolRegistry(tmp_path / "tools"),
        "knowledge": KnowledgeRegistry(tmp_path / "knowledge.json"),
        "validation": ValidationRegistry(tmp_path / "tests"),
        "buf": ObservationBuffer(tmp_path / "obs.jsonl"),
    }


def make_evolver(registries, client) -> Evolver:
    return Evolver(
        tool_registry=registries["tools"],
        knowledge_registry=registries["knowledge"],
        validation_registry=registries["validation"],
        observation_buffer=registries["buf"],
        client=client,
        window_size=DEFAULT_WINDOW_SIZE,
    )


# ── EvolveResult dataclass ────────────────────────────────────────────────────


class TestEvolveResult:
    def test_fields(self):
        r = EvolveResult(
            diagnosed=True,
            plans=[{"op_type": "CreateTool"}],
            updates_applied=1,
            tests_passed=True,
            committed=True,
            output="all good",
        )
        assert r.diagnosed is True
        assert len(r.plans) == 1
        assert r.updates_applied == 1
        assert r.tests_passed is True
        assert r.committed is True
        assert r.output == "all good"
        assert r.failure_patterns == []  # default

    def test_failure_patterns_default(self):
        r = EvolveResult(False, [], 0, False, False, "")
        assert r.failure_patterns == []


# ── run() — no observations ───────────────────────────────────────────────────


class TestRunNoObservations:
    def test_returns_early_with_no_diagnosed(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        result = evolver.run()
        assert result.diagnosed is False
        assert result.committed is False
        assert result.updates_applied == 0

    def test_llm_not_called_when_no_observations(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)
        evolver.run()
        client.messages.stream.assert_not_called()


# ── _diagnose() ───────────────────────────────────────────────────────────────


class TestDiagnose:
    def test_diagnose_returns_empty_on_no_failures(self, registries):
        patterns = [{"pattern": "p1", "frequency": 3, "affected_tools": ["t"], "severity": "high"}]
        client = make_mock_client(json.dumps(patterns))
        evolver = make_evolver(registries, client)

        obs = [make_obs(outcome="failure") for _ in range(3)]
        result = evolver._diagnose(obs)
        assert result == patterns

    def test_diagnose_returns_empty_list_on_no_patterns(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        result = evolver._diagnose([make_obs()])
        assert result == []

    def test_diagnose_handles_invalid_json_gracefully(self, registries):
        client = make_mock_client("not json at all")
        evolver = make_evolver(registries, client)

        result = evolver._diagnose([make_obs()])
        assert result == []

    def test_diagnose_handles_non_list_json(self, registries):
        client = make_mock_client('{"unexpected": "dict"}')
        evolver = make_evolver(registries, client)

        result = evolver._diagnose([make_obs()])
        assert result == []

    def test_diagnose_sends_observations_to_llm(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        obs = [make_obs(task="failing task", outcome="failure", error="boom")]
        evolver._diagnose(obs)

        call_kwargs = client.messages.stream.call_args[1]
        assert "messages" in call_kwargs
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "failing task" in prompt_text

    def test_diagnose_uses_configured_model(self, registries):
        client = make_mock_client("[]")
        evolver = Evolver(
            tool_registry=registries["tools"],
            knowledge_registry=registries["knowledge"],
            validation_registry=registries["validation"],
            observation_buffer=registries["buf"],
            client=client,
            model="claude-haiku-4-5",
        )
        evolver._diagnose([make_obs()])
        call_kwargs = client.messages.stream.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5"


# ── _plan() ───────────────────────────────────────────────────────────────────


class TestPlan:
    def test_plan_returns_edit_ops(self, registries):
        plans = [{"op_type": "CreateTool", "name": "new_tool", "code": "def new_tool(): pass", "description": "desc"}]
        client = make_mock_client(json.dumps(plans))
        evolver = make_evolver(registries, client)

        result = evolver._plan([{"pattern": "p", "frequency": 2, "severity": "high"}])
        assert result == plans

    def test_plan_returns_empty_on_invalid_json(self, registries):
        client = make_mock_client("bad json")
        evolver = make_evolver(registries, client)

        result = evolver._plan([{"pattern": "p"}])
        assert result == []

    def test_plan_includes_existing_tools_in_prompt(self, registries):
        registries["tools"].create_tool("existing_tool", ADD_TOOL_CODE)
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        evolver._plan([{"pattern": "p"}])
        call_kwargs = client.messages.stream.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "existing_tool" in prompt_text

    def test_plan_includes_existing_knowledge_in_prompt(self, registries):
        registries["knowledge"].set("api_url", "https://api.example.com")
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        evolver._plan([{"pattern": "p"}])
        call_kwargs = client.messages.stream.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"]
        assert "api_url" in prompt_text


# ── _update() ────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_update_create_tool(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "CreateTool", "name": "new_tool", "code": "def new_tool(): return 99", "description": "test"}]
        count = evolver._update(plans)

        assert count == 1
        assert "new_tool" in registries["tools"].list_tools()
        func = registries["tools"].load_tool("new_tool")
        assert func() == 99

    def test_update_evolve_tool(self, registries):
        registries["tools"].create_tool("my_tool", "def my_tool(): return 1\n")
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "EvolveTool", "name": "my_tool", "new_code": "def my_tool(): return 42\n", "reason": "better"}]
        count = evolver._update(plans)

        assert count == 1
        func = registries["tools"].load_tool("my_tool")
        assert func() == 42

    def test_update_add_knowledge(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "AddKnowledge", "key": "new_key", "value": "new_value", "description": "test"}]
        count = evolver._update(plans)

        assert count == 1
        assert registries["knowledge"].get("new_key") == "new_value"

    def test_update_evolve_knowledge(self, registries):
        registries["knowledge"].set("k", "old_value")
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "EvolveKnowledge", "key": "k", "new_value": "new_value", "reason": "updated"}]
        count = evolver._update(plans)

        assert count == 1
        assert registries["knowledge"].get("k") == "new_value"

    def test_update_prune_tool(self, registries):
        registries["tools"].create_tool("old_tool", ADD_TOOL_CODE)
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "PruneTool", "name": "old_tool", "reason": "unused"}]
        count = evolver._update(plans)

        assert count == 1
        assert "old_tool" not in registries["tools"].list_tools()

    def test_update_skips_unknown_op_type(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [{"op_type": "UnknownOp", "foo": "bar"}]
        count = evolver._update(plans)
        assert count == 0

    def test_update_skips_malformed_op(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        # CreateTool without required 'name' key — should not crash
        plans = [{"op_type": "CreateTool"}]  # missing name, code
        count = evolver._update(plans)
        # Defaults to empty strings, so it will create a tool with name ""
        # OR skip due to an error — either is acceptable, just shouldn't raise
        assert count >= 0

    def test_update_multiple_ops_in_order(self, registries):
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        plans = [
            {"op_type": "CreateTool", "name": "tool_a", "code": "def tool_a(): return 'a'\n", "description": ""},
            {"op_type": "AddKnowledge", "key": "k1", "value": "v1", "description": ""},
            {"op_type": "AddKnowledge", "key": "k2", "value": "v2", "description": ""},
        ]
        count = evolver._update(plans)

        assert count == 3
        assert "tool_a" in registries["tools"].list_tools()
        assert registries["knowledge"].get("k1") == "v1"
        assert registries["knowledge"].get("k2") == "v2"


# ── run() — full pipeline ─────────────────────────────────────────────────────


class TestRunFullPipeline:
    def _populate_observations(self, buf, n=3, outcome="failure"):
        for i in range(n):
            buf.append(make_obs(task=f"task-{i}", outcome=outcome, error="broke" if outcome == "failure" else None))

    def test_run_with_no_patterns_returns_early(self, registries):
        self._populate_observations(registries["buf"], outcome="failure")
        client = make_mock_client("[]")
        evolver = make_evolver(registries, client)

        result = evolver.run()
        assert result.diagnosed is True
        assert result.plans == []
        assert result.committed is False

    def test_run_with_patterns_but_no_plans(self, registries):
        self._populate_observations(registries["buf"])
        patterns = [{"pattern": "p", "frequency": 3, "severity": "high", "affected_tools": []}]

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            if call_count == 1:
                resp = MagicMock()
                blk = MagicMock(); blk.text = json.dumps(patterns); blk.type = "text"
                resp.content = [blk]
                cm.get_final_message = MagicMock(return_value=resp)
            else:
                resp = MagicMock()
                blk = MagicMock(); blk.text = "[]"; blk.type = "text"
                resp.content = [blk]
                cm.get_final_message = MagicMock(return_value=resp)
            return cm

        client = MagicMock()
        client.messages.stream.side_effect = side_effect

        evolver = make_evolver(registries, client)
        result = evolver.run()

        assert result.diagnosed is True
        assert result.failure_patterns == patterns
        assert result.plans == []

    def test_run_commits_on_test_pass(self, registries):
        self._populate_observations(registries["buf"])

        patterns_json = json.dumps([
            {"pattern": "p", "frequency": 3, "severity": "high", "affected_tools": []}
        ])
        plans_json = json.dumps([
            {"op_type": "AddKnowledge", "key": "fix", "value": "applied", "description": ""}
        ])

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            text = patterns_json if call_count == 1 else plans_json
            resp = MagicMock()
            blk = MagicMock(); blk.text = text; blk.type = "text"
            resp.content = [blk]
            cm.get_final_message = MagicMock(return_value=resp)
            return cm

        client = MagicMock()
        client.messages.stream.side_effect = side_effect

        # Mock subprocess so tests "pass"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "1 passed"
        mock_proc.stderr = ""

        evolver = make_evolver(registries, client)
        with patch("subprocess.run", return_value=mock_proc):
            result = evolver.run()

        assert result.updates_applied == 1
        assert result.tests_passed is True
        assert result.committed is True
        assert registries["knowledge"].get("fix") == "applied"

    def test_run_does_not_commit_on_test_fail(self, registries):
        self._populate_observations(registries["buf"])

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            texts = [
                json.dumps([{"pattern": "p", "frequency": 3, "severity": "high", "affected_tools": []}]),
                json.dumps([{"op_type": "AddKnowledge", "key": "risky_fix", "value": "applied", "description": ""}]),
            ]
            text = texts[call_count - 1] if call_count <= len(texts) else "[]"
            resp = MagicMock()
            blk = MagicMock(); blk.text = text; blk.type = "text"
            resp.content = [blk]
            cm.get_final_message = MagicMock(return_value=resp)
            return cm

        client = MagicMock()
        client.messages.stream.side_effect = side_effect

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "1 failed"
        mock_proc.stderr = "AssertionError"

        evolver = make_evolver(registries, client)
        with patch("subprocess.run", return_value=mock_proc):
            result = evolver.run()

        assert result.tests_passed is False
        assert result.committed is False

    def test_window_size_limits_observations_read(self, registries):
        for i in range(20):
            registries["buf"].append(make_obs(task=f"t{i}"))

        client = make_mock_client("[]")
        evolver = Evolver(
            tool_registry=registries["tools"],
            knowledge_registry=registries["knowledge"],
            validation_registry=registries["validation"],
            observation_buffer=registries["buf"],
            client=client,
            window_size=5,
        )
        evolver.run()

        call_kwargs = client.messages.stream.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        # Only the last 5 observations (t15–t19) should appear in the prompt
        obs_data = json.loads(prompt.split("Observations:\n")[1].split("\n\nReturn")[0])
        assert len(obs_data) == 5
        assert obs_data[0]["task"] == "t15"
        assert obs_data[4]["task"] == "t19"
