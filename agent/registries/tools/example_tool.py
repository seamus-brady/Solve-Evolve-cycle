"""
Example tool: echo a greeting message.

Demonstrates the minimal structure of a registered tool:
a single callable with a descriptive name and docstring.
"""


def example_tool(message: str = "Hello") -> str:
    """Return a greeting string from the tool.

    Args:
        message: The greeting text to wrap.
    """
    return f"Tool says: {message}"
