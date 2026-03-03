"""Tests for agent/registry.py"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.registry import KnowledgeRegistry, ValidationResult, ToolRegistry, ValidationRegistry


# ── KnowledgeRegistry ─────────────────────────────────────────────────────────


class TestKnowledgeRegistry:
    @pytest.fixture
    def reg(self, tmp_path) -> KnowledgeRegistry:
        return KnowledgeRegistry(tmp_path / "knowledge.json")

    def test_initially_empty(self, reg):
        assert reg.list_keys() == []

    def test_get_nonexistent_key_returns_none(self, reg):
        assert reg.get("missing") is None

    def test_set_and_get(self, reg):
        reg.set("api_url", "https://example.com")
        assert reg.get("api_url") == "https://example.com"

    def test_set_various_value_types(self, reg):
        reg.set("count", 42)
        reg.set("flag", True)
        reg.set("items", [1, 2, 3])
        reg.set("nested", {"a": {"b": 1}})

        assert reg.get("count") == 42
        assert reg.get("flag") is True
        assert reg.get("items") == [1, 2, 3]
        assert reg.get("nested") == {"a": {"b": 1}}

    def test_list_keys(self, reg):
        reg.set("a", 1)
        reg.set("b", 2)
        assert sorted(reg.list_keys()) == ["a", "b"]

    def test_delete_existing_key(self, reg):
        reg.set("x", "val")
        deleted = reg.delete("x")
        assert deleted is True
        assert reg.get("x") is None

    def test_delete_nonexistent_key(self, reg):
        deleted = reg.delete("ghost")
        assert deleted is False

    def test_delete_removes_from_list(self, reg):
        reg.set("keep", 1)
        reg.set("remove", 2)
        reg.delete("remove")
        assert "remove" not in reg.list_keys()
        assert "keep" in reg.list_keys()

    def test_overwrite_updates_value(self, reg):
        reg.set("k", "old")
        reg.set("k", "new")
        assert reg.get("k") == "new"

    def test_versioning_increments(self, reg):
        reg.set("k", "v1")
        d1 = reg.to_dict()
        v1 = d1["entries"]["k"]["version"]

        reg.set("k", "v2")
        d2 = reg.to_dict()
        v2 = d2["entries"]["k"]["version"]

        assert v2 == v1 + 1

    def test_registry_level_version_increments(self, reg):
        reg.set("a", 1)
        v1 = reg.to_dict()["version"]
        reg.set("b", 2)
        v2 = reg.to_dict()["version"]
        assert v2 > v1

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "k.json"
        r1 = KnowledgeRegistry(path)
        r1.set("key", "persisted_value")

        r2 = KnowledgeRegistry(path)
        assert r2.get("key") == "persisted_value"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "k.json"
        reg = KnowledgeRegistry(path)
        reg.set("x", 1)
        assert path.exists()

    def test_description_and_provenance_stored(self, reg):
        reg.set("k", "v", description="my desc", provenance="test")
        raw = reg.to_dict()["entries"]["k"]
        assert raw["description"] == "my desc"
        assert raw["provenance"] == "test"

    def test_to_dict_contains_entries_key(self, reg):
        reg.set("k", "v")
        d = reg.to_dict()
        assert "entries" in d
        assert "version" in d


# ── ToolRegistry ──────────────────────────────────────────────────────────────


SIMPLE_TOOL_CODE = """\
def simple_tool(x: int = 0) -> int:
    return x * 2
"""

FAILING_TOOL_CODE = """\
def failing_tool():
    raise RuntimeError("always fails")
"""


class TestToolRegistry:
    @pytest.fixture
    def reg(self, tmp_path) -> ToolRegistry:
        return ToolRegistry(tmp_path / "tools")

    def test_creates_directory(self, tmp_path):
        reg = ToolRegistry(tmp_path / "tools")
        assert (tmp_path / "tools").is_dir()

    def test_list_tools_initially_empty(self, reg):
        assert reg.list_tools() == []

    def test_create_tool_adds_file(self, reg):
        reg.create_tool("my_tool", SIMPLE_TOOL_CODE)
        assert reg.get_tool_path("my_tool").exists()

    def test_get_tool_code_returns_source(self, reg):
        reg.create_tool("t", SIMPLE_TOOL_CODE)
        code = reg.get_tool_code("t")
        assert "simple_tool" in code

    def test_get_tool_code_nonexistent_returns_none(self, reg):
        assert reg.get_tool_code("ghost") is None

    def test_list_tools_after_create(self, reg):
        reg.create_tool("alpha", SIMPLE_TOOL_CODE)
        reg.create_tool("beta", SIMPLE_TOOL_CODE)
        assert sorted(reg.list_tools()) == ["alpha", "beta"]

    def test_list_tools_excludes_init(self, reg):
        (reg.tools_dir / "__init__.py").write_text("")
        reg.create_tool("real_tool", SIMPLE_TOOL_CODE)
        assert "__init__" not in reg.list_tools()

    def test_delete_tool_removes_file(self, reg):
        reg.create_tool("t", SIMPLE_TOOL_CODE)
        deleted = reg.delete_tool("t")
        assert deleted is True
        assert not reg.get_tool_path("t").exists()

    def test_delete_nonexistent_tool(self, reg):
        assert reg.delete_tool("ghost") is False

    def test_delete_removes_from_list(self, reg):
        reg.create_tool("keep", SIMPLE_TOOL_CODE)
        reg.create_tool("del", SIMPLE_TOOL_CODE)
        reg.delete_tool("del")
        assert "del" not in reg.list_tools()
        assert "keep" in reg.list_tools()

    def test_load_tool_returns_callable(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        func = reg.load_tool("simple_tool")
        assert callable(func)

    def test_load_tool_executes_correctly(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        func = reg.load_tool("simple_tool")
        assert func(x=5) == 10

    def test_load_tool_nonexistent_returns_none(self, reg):
        assert reg.load_tool("ghost") is None

    def test_load_all_tools(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        tools = reg.load_all_tools()
        assert "simple_tool" in tools
        assert callable(tools["simple_tool"])

    def test_create_overwrites_existing(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        new_code = "def simple_tool(x: int = 0) -> int:\n    return x + 100\n"
        reg.create_tool("simple_tool", new_code)
        func = reg.load_tool("simple_tool")
        assert func(x=0) == 100

    def test_cache_invalidated_on_overwrite(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        reg.load_tool("simple_tool")  # warm cache

        new_code = "def simple_tool(x=0):\n    return x + 99\n"
        reg.create_tool("simple_tool", new_code)
        func = reg.load_tool("simple_tool")
        assert func(x=1) == 100  # 1 + 99

    def test_cache_invalidated_on_delete(self, reg):
        reg.create_tool("simple_tool", SIMPLE_TOOL_CODE)
        reg.load_tool("simple_tool")  # warm cache
        reg.delete_tool("simple_tool")
        assert reg.load_tool("simple_tool") is None

    def test_load_tool_fallback_to_first_callable(self, reg):
        code = "def my_func(): return 'fallback'\n"
        reg.create_tool("other_name", code)
        func = reg.load_tool("other_name")
        assert func is not None
        assert func() == "fallback"


# ── ValidationRegistry ────────────────────────────────────────────────────────


PASSING_TEST_CODE = """\
def test_always_passes():
    assert 1 + 1 == 2
"""

FAILING_TEST_CODE = """\
def test_always_fails():
    assert False, "intentional failure"
"""


class TestValidationRegistry:
    @pytest.fixture
    def reg(self, tmp_path) -> ValidationRegistry:
        return ValidationRegistry(tmp_path / "tests")

    def test_creates_directory(self, tmp_path):
        reg = ValidationRegistry(tmp_path / "tests")
        assert (tmp_path / "tests").is_dir()

    def test_create_test_writes_file(self, reg):
        reg.create_test("my_tool", PASSING_TEST_CODE)
        assert reg.get_test_path("my_tool").exists()

    def test_get_test_code_returns_source(self, reg):
        reg.create_test("t", PASSING_TEST_CODE)
        code = reg.get_test_code("t")
        assert "test_always_passes" in code

    def test_get_test_code_nonexistent_returns_none(self, reg):
        assert reg.get_test_code("ghost") is None

    def test_list_tests(self, reg):
        reg.create_test("tool_a", PASSING_TEST_CODE)
        reg.create_test("tool_b", PASSING_TEST_CODE)
        assert sorted(reg.list_tests()) == ["tool_a", "tool_b"]

    def test_delete_test_removes_file(self, reg):
        reg.create_test("t", PASSING_TEST_CODE)
        deleted = reg.delete_test("t")
        assert deleted is True
        assert not reg.get_test_path("t").exists()

    def test_delete_nonexistent_test(self, reg):
        assert reg.delete_test("ghost") is False

    def test_run_tests_success(self, reg):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "1 passed"
        mock_proc.stderr = ""

        reg.create_test("my_tool", PASSING_TEST_CODE)
        with patch("subprocess.run", return_value=mock_proc):
            result = reg.run_tests("my_tool")

        assert result.passed is True
        assert result.return_code == 0

    def test_run_tests_failure(self, reg):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = "1 failed"
        mock_proc.stderr = ""

        reg.create_test("my_tool", FAILING_TEST_CODE)
        with patch("subprocess.run", return_value=mock_proc):
            result = reg.run_tests("my_tool")

        assert result.passed is False
        assert result.return_code == 1

    def test_run_tests_no_test_file_returns_failure(self, reg):
        result = reg.run_tests("nonexistent_tool")
        assert result.passed is False
        assert "nonexistent_tool" in result.output

    def test_run_tests_all_uses_tests_dir(self, reg):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "all passed"
        mock_proc.stderr = ""

        reg.create_test("tool_a", PASSING_TEST_CODE)
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            result = reg.run_tests()

        assert result.passed is True
        # Confirm the call used the tests directory (not a specific file)
        call_args = mock_run.call_args[0][0]
        assert str(reg.tests_dir) in call_args

    def test_validation_result_dataclass(self):
        tr = ValidationResult(passed=True, output="ok", return_code=0)
        assert tr.passed is True
        assert tr.output == "ok"
        assert tr.return_code == 0
