"""Let the tests run from a source checkout without installing the package.

Puts ``src/`` on ``sys.path`` so ``import specbeat`` resolves. After
``pip install -e .`` this is a no-op: the installed path takes precedence.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
