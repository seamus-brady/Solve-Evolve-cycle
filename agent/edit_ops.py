"""
Typed edit operations for the Evolver pipeline.

The evolver maps failure patterns to one of these typed operations,
giving the LLM cleaner structure than open-ended "improve yourself."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union


@dataclass
class CreateTool:
    """Create a new tool in the Tool registry."""

    op_type: Literal["CreateTool"] = field(default="CreateTool", init=False)
    name: str = ""
    code: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "name": self.name,
            "code": self.code,
            "description": self.description,
        }


@dataclass
class EvolveTool:
    """Replace the code of an existing tool."""

    op_type: Literal["EvolveTool"] = field(default="EvolveTool", init=False)
    name: str = ""
    new_code: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "name": self.name,
            "new_code": self.new_code,
            "reason": self.reason,
        }


@dataclass
class AddKnowledge:
    """Add a new entry to the Knowledge registry."""

    op_type: Literal["AddKnowledge"] = field(default="AddKnowledge", init=False)
    key: str = ""
    value: Any = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "key": self.key,
            "value": self.value,
            "description": self.description,
        }


@dataclass
class EvolveKnowledge:
    """Update an existing knowledge entry."""

    op_type: Literal["EvolveKnowledge"] = field(default="EvolveKnowledge", init=False)
    key: str = ""
    new_value: Any = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "key": self.key,
            "new_value": self.new_value,
            "reason": self.reason,
        }


@dataclass
class PruneTool:
    """Remove a tool from the Tool registry."""

    op_type: Literal["PruneTool"] = field(default="PruneTool", init=False)
    name: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "name": self.name,
            "reason": self.reason,
        }


# Union type for all edit operations
EditOperation = Union[CreateTool, EvolveTool, AddKnowledge, EvolveKnowledge, PruneTool]

OP_TYPES = {
    "CreateTool": CreateTool,
    "EvolveTool": EvolveTool,
    "AddKnowledge": AddKnowledge,
    "EvolveKnowledge": EvolveKnowledge,
    "PruneTool": PruneTool,
}


def from_dict(d: Dict[str, Any]) -> Optional[EditOperation]:
    """Deserialize an edit operation from a dict (e.g. from LLM JSON output)."""
    op_type = d.get("op_type")
    cls = OP_TYPES.get(op_type)
    if cls is None:
        return None

    # Build kwargs without op_type (it's a non-init field)
    kwargs = {k: v for k, v in d.items() if k != "op_type"}
    try:
        return cls(**kwargs)
    except TypeError:
        return None
