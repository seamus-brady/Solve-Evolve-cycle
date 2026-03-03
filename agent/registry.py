"""
Three Persistent Registries.

| Registry      | Contents                                             |
|---------------|------------------------------------------------------|
| Knowledge (K) | Schemas, workflows, interface contracts, examples    |
| Tool (T)      | Executable Python functions with signatures          |
| Validation (V)| Unit tests + regression suites, run before commits   |

All three live on disk, are version-tracked, and are read-only during
the Solve phase but writable during the Evolve phase.
"""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Knowledge Registry ────────────────────────────────────────────────────────


class KnowledgeRegistry:
    """
    Versioned key-value store of facts, schemas, and examples.

    Backed by a single JSON file. Each entry records its value, a description,
    provenance metadata, and a per-entry version counter.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Any] = {}
        self._load()

    # ── public API ────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return the stored value for key, or None."""
        entry = self._data.get("entries", {}).get(key)
        if entry is None:
            return None
        return entry.get("value") if isinstance(entry, dict) else entry

    def set(
        self,
        key: str,
        value: Any,
        description: str = "",
        provenance: str = "",
    ) -> None:
        """Create or update a knowledge entry and persist."""
        entries = self._data.setdefault("entries", {})
        prev_version = 0
        if key in entries and isinstance(entries[key], dict):
            prev_version = entries[key].get("version", 0)

        entries[key] = {
            "value": value,
            "description": description,
            "provenance": provenance,
            "updated_at": time.time(),
            "version": prev_version + 1,
        }
        self._data["version"] = self._data.get("version", 0) + 1
        self._save()

    def delete(self, key: str) -> bool:
        """Remove a key; returns True if it existed."""
        entries = self._data.get("entries", {})
        if key in entries:
            del entries[key]
            self._save()
            return True
        return False

    def list_keys(self) -> List[str]:
        """Return all knowledge keys."""
        return list(self._data.get("entries", {}).keys())

    def to_dict(self) -> Dict[str, Any]:
        """Return the raw backing dict (for inspection/testing)."""
        return dict(self._data)

    # ── internal ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        else:
            self._data = {"version": 0, "entries": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)


# ── Tool Registry ─────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Manages executable Python tool files — one tool per file.

    Tools are stored as plain .py files. The registry supports dynamic loading
    via exec() so tools can be evolved and reloaded without restarting.
    """

    def __init__(self, tools_dir: Path) -> None:
        self.tools_dir = tools_dir
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Callable] = {}

    # ── public API ────────────────────────────────────────────────────────

    def create_tool(self, name: str, code: str, description: str = "") -> None:
        """Write (or overwrite) a tool's source file."""
        self.get_tool_path(name).write_text(code)
        self._cache.pop(name, None)  # Invalidate cached function

    def get_tool_code(self, name: str) -> Optional[str]:
        """Return the source code of a tool, or None if absent."""
        path = self.get_tool_path(name)
        return path.read_text() if path.exists() else None

    def delete_tool(self, name: str) -> bool:
        """Delete a tool file; returns True if it existed."""
        path = self.get_tool_path(name)
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            return True
        return False

    def list_tools(self) -> List[str]:
        """Return names of all registered tools (filename stems)."""
        return sorted(
            p.stem
            for p in self.tools_dir.glob("*.py")
            if p.stem not in ("__init__",) and not p.stem.startswith("_")
        )

    def load_tool(self, name: str) -> Optional[Callable]:
        """
        Dynamically load and return the main callable from a tool file.

        Looks for a function named after the tool, then 'run', then 'execute',
        then the first non-private callable in the file's namespace.
        """
        if name in self._cache:
            return self._cache[name]

        path = self.get_tool_path(name)
        if not path.exists():
            return None

        namespace: Dict[str, Any] = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)

        func = (
            namespace.get(name)
            or namespace.get("run")
            or namespace.get("execute")
        )
        if func is None:
            # Fallback: first non-private callable defined in the file
            for fname, fval in namespace.items():
                if callable(fval) and not fname.startswith("_"):
                    func = fval
                    break

        if func is not None and callable(func):
            self._cache[name] = func
        return func

    def load_all_tools(self) -> Dict[str, Callable]:
        """Load all registered tools and return {name: callable}."""
        result = {}
        for name in self.list_tools():
            func = self.load_tool(name)
            if func is not None:
                result[name] = func
        return result

    def get_tool_path(self, name: str) -> Path:
        return self.tools_dir / f"{name}.py"


# ── Validation Registry ───────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Result of running the validation suite."""

    passed: bool
    output: str
    return_code: int


class ValidationRegistry:
    """
    Manages pytest test files — one test file per tool.

    The Evolver runs these tests against proposed changes; only artifacts
    that pass all tests are committed ("never commit without tests passing").
    """

    def __init__(self, tests_dir: Path) -> None:
        self.tests_dir = tests_dir
        self.tests_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def create_test(self, tool_name: str, test_code: str) -> None:
        """Write (or overwrite) a test file for a tool."""
        self.get_test_path(tool_name).write_text(test_code)

    def get_test_code(self, tool_name: str) -> Optional[str]:
        """Return test source code, or None if absent."""
        path = self.get_test_path(tool_name)
        return path.read_text() if path.exists() else None

    def delete_test(self, tool_name: str) -> bool:
        """Delete a test file; returns True if it existed."""
        path = self.get_test_path(tool_name)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_tests(self) -> List[str]:
        """Return tool names that have associated test files."""
        return sorted(
            p.stem[len("test_"):]
            for p in self.tests_dir.glob("test_*.py")
        )

    def run_tests(self, tool_name: Optional[str] = None) -> ValidationResult:
        """
        Run the validation suite with pytest.

        If tool_name is given, runs only that tool's tests.
        Returns a TestResult indicating pass/fail and captured output.
        """
        if tool_name is not None:
            path = self.get_test_path(tool_name)
            if not path.exists():
                return ValidationResult(
                    passed=False,
                    output=f"No tests found for tool '{tool_name}'.",
                    return_code=1,
                )
            test_path = str(path)
        else:
            test_path = str(self.tests_dir)

        proc = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
        )
        return ValidationResult(
            passed=proc.returncode == 0,
            output=proc.stdout + proc.stderr,
            return_code=proc.returncode,
        )

    def get_test_path(self, tool_name: str) -> Path:
        return self.tests_dir / f"test_{tool_name}.py"
