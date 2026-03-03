"""
Validation tests for the example_tool.

These live in the Validation registry (V) and are run by the Evolver's
verify step before committing any changes to the Tool registry.
"""

import sys
from pathlib import Path

# Add the tools directory to sys.path so we can import tools directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from example_tool import example_tool  # noqa: E402


def test_example_tool_default_message():
    result = example_tool()
    assert result == "Tool says: Hello"


def test_example_tool_custom_message():
    result = example_tool(message="World")
    assert result == "Tool says: World"


def test_example_tool_returns_string():
    result = example_tool(message="test")
    assert isinstance(result, str)


def test_example_tool_prefix():
    result = example_tool(message="anything")
    assert result.startswith("Tool says: ")
