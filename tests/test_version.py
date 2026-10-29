"""One version string, three declarations: they must agree.

The version is written by hand in ``pyproject.toml``, ``CITATION.cff`` and
``specbeat.__version__``, on purpose: the package must import from a plain
``src`` checkout without being installed, so ``importlib.metadata`` is not an
option, and a build-time stamp would add tooling to a three-dependency package.
This test is the single guard that keeps the three copies equal.
"""
import pathlib
import re

import specbeat

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_consistent_everywhere():
    pyproject = (ROOT / "pyproject.toml").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    v_py = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    v_cff = re.search(r"^version: (\S+)", citation, re.M).group(1)
    assert specbeat.__version__ == v_py == v_cff
