"""specbeat: detecting faint spectroscopic companions in Fourier space.

A small, simulation-only package accompanying the paper *Faint Companions to
Hot Stars*. It depends on numpy, scipy and matplotlib, and on nothing else.

Submodules
----------
kernels    -- Fourier-space broadening kernels (plain floats).
simulate   -- synthetic single- and double-lined spectra, plus noise.
amplitude  -- the log-|FT| observable, its design matrix, and bounded WLS.
phase      -- the phase observable arg(S) and its fit (eq. 14).
detect     -- velocity scan, single-star fit, detection statistic.
nulls      -- the alpha >= 0 boundary null: thresholds, p-values, upper limits.
power      -- non-centrality, detection power, Fisher information.
mc         -- Monte-Carlo nulls and injection-recovery.
plotting   -- the matplotlib settings used for the paper figures.
"""

from . import kernels
from . import simulate
from . import amplitude
from . import phase
from . import detect
from . import nulls
from . import power
from . import mc
from . import plotting

__all__ = [
    "kernels",
    "simulate",
    "amplitude",
    "phase",
    "detect",
    "nulls",
    "power",
    "mc",
    "plotting",
]

__version__ = "0.1.0"
