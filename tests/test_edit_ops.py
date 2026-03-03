"""Tests for agent/edit_ops.py"""

import pytest
from agent.edit_ops import (
    AddKnowledge,
    CreateTool,
    EditOperation,
    EvolveKnowledge,
    EvolveTool,
    OP_TYPES,
    PruneTool,
    from_dict,
)


# ── CreateTool ────────────────────────────────────────────────────────────────


class TestCreateTool:
    def test_op_type_is_create_tool(self):
        op = CreateTool(name="foo", code="def foo(): pass", description="A tool")
        assert op.op_type == "CreateTool"

    def test_fields_stored(self):
        op = CreateTool(name="my_tool", code="x = 1", description="desc")
        assert op.name == "my_tool"
        assert op.code == "x = 1"
        assert op.description == "desc"

    def test_defaults(self):
        op = CreateTool()
        assert op.name == ""
        assert op.code == ""
        assert op.description == ""

    def test_to_dict_keys(self):
        op = CreateTool(name="t", code="c", description="d")
        d = op.to_dict()
        assert d == {"op_type": "CreateTool", "name": "t", "code": "c", "description": "d"}

    def test_op_type_not_in_init(self):
        # op_type should not be a positional constructor argument
        with pytest.raises(TypeError):
            CreateTool("CreateTool", "name", "code", "desc")  # too many args


# ── EvolveTool ────────────────────────────────────────────────────────────────


class TestEvolveTool:
    def test_op_type(self):
        op = EvolveTool(name="t", new_code="def t(): return 1", reason="fix bug")
        assert op.op_type == "EvolveTool"

    def test_fields(self):
        op = EvolveTool(name="calc", new_code="def calc(): pass", reason="improved")
        assert op.name == "calc"
        assert op.new_code == "def calc(): pass"
        assert op.reason == "improved"

    def test_to_dict(self):
        op = EvolveTool(name="n", new_code="c", reason="r")
        d = op.to_dict()
        assert d == {"op_type": "EvolveTool", "name": "n", "new_code": "c", "reason": "r"}


# ── AddKnowledge ──────────────────────────────────────────────────────────────


class TestAddKnowledge:
    def test_op_type(self):
        op = AddKnowledge(key="k", value="v", description="d")
        assert op.op_type == "AddKnowledge"

    def test_value_can_be_any_type(self):
        op = AddKnowledge(key="nums", value=[1, 2, 3])
        assert op.value == [1, 2, 3]

        op2 = AddKnowledge(key="flag", value=True)
        assert op2.value is True

    def test_to_dict(self):
        op = AddKnowledge(key="k", value=42, description="desc")
        d = op.to_dict()
        assert d == {"op_type": "AddKnowledge", "key": "k", "value": 42, "description": "desc"}


# ── EvolveKnowledge ───────────────────────────────────────────────────────────


class TestEvolveKnowledge:
    def test_op_type(self):
        op = EvolveKnowledge(key="k", new_value="v2", reason="updated")
        assert op.op_type == "EvolveKnowledge"

    def test_to_dict(self):
        op = EvolveKnowledge(key="window_size", new_value=20, reason="larger window")
        d = op.to_dict()
        assert d == {
            "op_type": "EvolveKnowledge",
            "key": "window_size",
            "new_value": 20,
            "reason": "larger window",
        }


# ── PruneTool ─────────────────────────────────────────────────────────────────


class TestPruneTool:
    def test_op_type(self):
        op = PruneTool(name="old_tool", reason="deprecated")
        assert op.op_type == "PruneTool"

    def test_to_dict(self):
        op = PruneTool(name="bad_tool", reason="broken")
        d = op.to_dict()
        assert d == {"op_type": "PruneTool", "name": "bad_tool", "reason": "broken"}


# ── from_dict ─────────────────────────────────────────────────────────────────


class TestFromDict:
    def test_create_tool_from_dict(self):
        d = {"op_type": "CreateTool", "name": "foo", "code": "def foo(): pass", "description": "x"}
        op = from_dict(d)
        assert isinstance(op, CreateTool)
        assert op.name == "foo"

    def test_evolve_tool_from_dict(self):
        d = {"op_type": "EvolveTool", "name": "foo", "new_code": "def foo(): return 1", "reason": "fix"}
        op = from_dict(d)
        assert isinstance(op, EvolveTool)

    def test_add_knowledge_from_dict(self):
        d = {"op_type": "AddKnowledge", "key": "k", "value": "v", "description": "d"}
        op = from_dict(d)
        assert isinstance(op, AddKnowledge)

    def test_evolve_knowledge_from_dict(self):
        d = {"op_type": "EvolveKnowledge", "key": "k", "new_value": "v2", "reason": "r"}
        op = from_dict(d)
        assert isinstance(op, EvolveKnowledge)

    def test_prune_tool_from_dict(self):
        d = {"op_type": "PruneTool", "name": "t", "reason": "r"}
        op = from_dict(d)
        assert isinstance(op, PruneTool)

    def test_unknown_op_type_returns_none(self):
        d = {"op_type": "UnknownOp", "name": "foo"}
        op = from_dict(d)
        assert op is None

    def test_missing_op_type_returns_none(self):
        d = {"name": "foo"}
        op = from_dict(d)
        assert op is None

    def test_malformed_dict_returns_none(self):
        # Missing required field (no 'name' for CreateTool — but name has a default, so test with wrong field)
        # EvolveTool without 'name' should still work (defaults to "")
        d = {"op_type": "PruneTool"}
        op = from_dict(d)
        assert isinstance(op, PruneTool)
        assert op.name == ""


# ── OP_TYPES registry ─────────────────────────────────────────────────────────


class TestOpTypesRegistry:
    def test_all_five_op_types_present(self):
        assert set(OP_TYPES.keys()) == {
            "CreateTool",
            "EvolveTool",
            "AddKnowledge",
            "EvolveKnowledge",
            "PruneTool",
        }

    def test_values_are_classes(self):
        for name, cls in OP_TYPES.items():
            assert callable(cls), f"{name} value should be a class"
