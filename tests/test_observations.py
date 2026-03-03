"""Tests for agent/observations.py"""

import json
import time
from pathlib import Path

import pytest

from agent.observations import Observation, ObservationBuffer, ToolCall


# ── ToolCall ──────────────────────────────────────────────────────────────────


class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(tool_name="my_tool", input={"x": 1}, output="42")
        assert tc.tool_name == "my_tool"
        assert tc.input == {"x": 1}
        assert tc.output == "42"
        assert tc.error is None

    def test_with_error(self):
        tc = ToolCall(tool_name="t", input={}, output="", error="TypeError: bad arg")
        assert tc.error == "TypeError: bad arg"

    def test_input_can_be_nested(self):
        tc = ToolCall(tool_name="t", input={"nested": {"a": [1, 2]}}, output="ok")
        assert tc.input["nested"]["a"] == [1, 2]


# ── Observation ───────────────────────────────────────────────────────────────


class TestObservation:
    def test_basic_creation(self):
        obs = Observation(task="do something", tool_calls=[], outcome="success")
        assert obs.task == "do something"
        assert obs.outcome == "success"
        assert obs.tool_calls == []
        assert obs.error is None
        assert obs.episode_id is None

    def test_with_tool_calls(self):
        tc = ToolCall(tool_name="foo", input={"k": "v"}, output="result")
        obs = Observation(task="t", tool_calls=[tc], outcome="success")
        assert len(obs.tool_calls) == 1
        assert obs.tool_calls[0].tool_name == "foo"

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        obs = Observation(task="t", tool_calls=[], outcome="success")
        after = time.time()
        assert before <= obs.timestamp <= after

    def test_failure_outcome(self):
        obs = Observation(
            task="t", tool_calls=[], outcome="failure", error="something went wrong"
        )
        assert obs.outcome == "failure"
        assert obs.error == "something went wrong"


# ── ObservationBuffer ─────────────────────────────────────────────────────────


class TestObservationBuffer:
    @pytest.fixture
    def log_path(self, tmp_path) -> Path:
        return tmp_path / "obs.jsonl"

    @pytest.fixture
    def buf(self, log_path) -> ObservationBuffer:
        return ObservationBuffer(log_path)

    def _make_obs(self, task="task", outcome="success", error=None) -> Observation:
        return Observation(
            task=task,
            tool_calls=[ToolCall(tool_name="t", input={}, output="out", error=error)],
            outcome=outcome,
            error=error,
            timestamp=1000.0,
            episode_id="ep-1",
        )

    # ── append ────────────────────────────────────────────────────────────

    def test_append_creates_file(self, buf, log_path):
        assert not log_path.exists()
        buf.append(self._make_obs())
        assert log_path.exists()

    def test_append_writes_valid_json_line(self, buf, log_path):
        buf.append(self._make_obs(task="my task"))
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["task"] == "my task"

    def test_append_multiple_entries(self, buf, log_path):
        for i in range(5):
            buf.append(self._make_obs(task=f"task-{i}"))
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 5

    # ── read_all ──────────────────────────────────────────────────────────

    def test_read_all_empty_when_no_file(self, buf):
        assert buf.read_all() == []

    def test_read_all_returns_all_observations(self, buf):
        for i in range(3):
            buf.append(self._make_obs(task=f"t{i}"))
        obs_list = buf.read_all()
        assert len(obs_list) == 3
        assert [o.task for o in obs_list] == ["t0", "t1", "t2"]

    def test_read_all_preserves_tool_calls(self, buf):
        obs = self._make_obs()
        buf.append(obs)
        loaded = buf.read_all()
        assert len(loaded[0].tool_calls) == 1
        assert loaded[0].tool_calls[0].tool_name == "t"

    # ── read_last_n ───────────────────────────────────────────────────────

    def test_read_last_n_fewer_than_n(self, buf):
        for i in range(3):
            buf.append(self._make_obs(task=f"t{i}"))
        result = buf.read_last_n(10)
        assert len(result) == 3

    def test_read_last_n_returns_last(self, buf):
        for i in range(7):
            buf.append(self._make_obs(task=f"t{i}"))
        result = buf.read_last_n(3)
        assert len(result) == 3
        assert result[0].task == "t4"
        assert result[2].task == "t6"

    def test_read_last_n_empty_file(self, buf):
        assert buf.read_last_n(5) == []

    def test_read_last_n_exact(self, buf):
        for i in range(5):
            buf.append(self._make_obs(task=f"t{i}"))
        result = buf.read_last_n(5)
        assert len(result) == 5

    # ── serialization roundtrip ───────────────────────────────────────────

    def test_roundtrip_preserves_fields(self, buf):
        original = Observation(
            task="roundtrip test",
            tool_calls=[
                ToolCall(tool_name="calc", input={"a": 1, "b": 2}, output="3", error=None),
                ToolCall(tool_name="fetch", input={"url": "x"}, output="data", error="timeout"),
            ],
            outcome="failure",
            error="calc failed",
            timestamp=999.5,
            episode_id="ep-abc",
        )
        buf.append(original)
        [loaded] = buf.read_all()

        assert loaded.task == "roundtrip test"
        assert loaded.outcome == "failure"
        assert loaded.error == "calc failed"
        assert loaded.timestamp == 999.5
        assert loaded.episode_id == "ep-abc"
        assert len(loaded.tool_calls) == 2
        assert loaded.tool_calls[0].tool_name == "calc"
        assert loaded.tool_calls[1].error == "timeout"

    # ── count ─────────────────────────────────────────────────────────────

    def test_count_empty(self, buf):
        assert buf.count() == 0

    def test_count_after_appends(self, buf):
        for _ in range(4):
            buf.append(self._make_obs())
        assert buf.count() == 4

    # ── clear ─────────────────────────────────────────────────────────────

    def test_clear_removes_file(self, buf, log_path):
        buf.append(self._make_obs())
        buf.clear()
        assert not log_path.exists()

    def test_clear_on_empty_buffer_is_noop(self, buf):
        buf.clear()  # should not raise

    # ── parent directory creation ─────────────────────────────────────────

    def test_creates_parent_directories(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "obs.jsonl"
        buf = ObservationBuffer(nested_path)
        buf.append(self._make_obs())
        assert nested_path.exists()
