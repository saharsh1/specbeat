"""The paper plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from specbeat import plotting


def test_set_paper_style_applies_rcparams():
    plt.rcParams["axes.grid"] = True
    applied = plotting.set_paper_style()
    assert plt.rcParams["axes.grid"] is False
    assert plt.rcParams["xtick.direction"] == "in"
    assert applied == plotting.PAPER_RCPARAMS


@pytest.mark.parametrize(
    "loc", ["upper right", "upper left", "lower right", "lower left"])
def test_regime_label_places_text_in_each_corner(loc):
    fig, ax = plt.subplots()
    text = plotting.regime_label(ax, "regime A", loc=loc)
    assert text.get_text() == "regime A"
    assert text in ax.texts
    plt.close(fig)


def test_regime_label_rejects_unknown_corner():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="loc must be one of"):
        plotting.regime_label(ax, "x", loc="middle")
    plt.close(fig)
