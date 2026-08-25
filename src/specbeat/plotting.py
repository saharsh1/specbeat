"""Matplotlib settings used for the figures in the paper.

Importing this module changes nothing; call :func:`set_paper_style` to apply the
rcParams. LaTeX rendering is off so the notebooks run without a TeX install.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["PAPER_RCPARAMS", "set_paper_style", "regime_label"]

#: rcParams for the paper figures: sans-serif, inward ticks, no grid.
PAPER_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 11.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.2,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
}

_CORNERS = {
    "upper right": (0.97, 0.97, "right", "top"),
    "upper left": (0.03, 0.97, "left", "top"),
    "lower right": (0.97, 0.03, "right", "bottom"),
    "lower left": (0.03, 0.03, "left", "bottom"),
}


def set_paper_style():
    """Apply :data:`PAPER_RCPARAMS` to the active matplotlib session.

    Returns
    -------
    dict
        The rcParams that were applied.
    """
    plt.rcParams.update(PAPER_RCPARAMS)
    return dict(PAPER_RCPARAMS)


def regime_label(ax, text, loc="upper right", fontsize=8):
    """Draw a boxed label in a corner of ``ax``, as in the paper's panels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    text : str
        Label text.
    loc : str
        One of ``"upper right"``, ``"upper left"``, ``"lower right"``,
        ``"lower left"``.
    fontsize : float
        Label font size (points).

    Returns
    -------
    matplotlib.text.Text
        The annotation, so the caller can restyle it.
    """
    try:
        x, y, ha, va = _CORNERS[loc]
    except KeyError:
        raise ValueError(
            f"loc must be one of {sorted(_CORNERS)}, got {loc!r}"
        ) from None
    return ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="k", lw=0.6),
    )
