"""One version string, three declarations: they must agree."""
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
