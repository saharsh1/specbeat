"""The dependency floor: numpy, scipy and matplotlib, and nothing else.

astropy is the one most likely to creep back in, since the research code this
was extracted from (``specbin``) uses it throughout. Both names are checked.

(a) Import every submodule in a FRESH subprocess and assert no ``sys.modules``
    key starts with a forbidden name — a subprocess, so that a sibling test
    which imported astropy cannot pollute the check.
(b) AST-scan every file under ``src/specbeat/`` for ``Import`` / ``ImportFrom``
    nodes naming one.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]          # repository root
_SRC = _PKG_ROOT / "src"
_SPECBEAT = _SRC / "specbeat"

_SUBMODULES = [
    "kernels", "simulate", "amplitude", "phase", "detect",
    "nulls", "power", "mc", "plotting",
]

FORBIDDEN = ("specbin", "astropy")


def _py_files():
    return sorted(_SPECBEAT.rglob("*.py"))


# --- (a) runtime import graph, checked in a clean subprocess -----------------
def test_no_forbidden_modules_imported_at_runtime():
    imports = "; ".join(f"import specbeat.{m}" for m in _SUBMODULES)
    code = (
        "import sys; import specbeat; "
        + imports
        + "; "
        + "bad = sorted(k for k in sys.modules "
        + "if k.split('.')[0] in ('specbin', 'astropy')); "
        + "print(repr(bad)); "
        + "sys.exit(1 if bad else 0)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, (
        f"forbidden modules imported by specbeat: {proc.stdout.strip()} "
        f"{proc.stderr.strip()}"
    )


# --- (b) static AST scan -----------------------------------------------------
@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.name)
def test_no_forbidden_imports_in_source(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN:
                offenders.append(f"from {node.module} import ...")
    assert not offenders, f"{path.name} imports forbidden modules: {offenders}"
