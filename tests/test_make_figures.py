import importlib.util
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@pytest.fixture(scope="module")
def mf():
    spec = importlib.util.spec_from_file_location("make_figures", FIGURES / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_split_labels_keeps_current_names(mf):
    assert mf._format_split_labels(["GR-SMILES", "GR-rxnsmi"]) == ["GR-SMILES", "GR-rxnsmi"]


def _trend(mf, order, means):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()
    means = np.array(means, dtype=float)
    ax.set_ylim(0, 2)
    mf._add_trend_labels(ax, np.arange(len(order)), means, np.zeros_like(means), order)
    return fig, ax


def test_trend_labels_two_reference_groups(mf):
    order = mf.SPLITS_ORDER_KINETICS
    assert order == [
        "Original",
        "GR-SMILES",
        "Mol-2D",
        "Mol-3D",
        "GR-seq",
        "Time",
        "P-seq",
        "P-struct",
    ]
    _, ax = _trend(mf, order, [[1, 1, 1.5, 0.5, 2, 3, 1, 4]])
    assert [t.get_text() for t in ax._trend_labels] == ["+50%", "-50%", "+50%", "-50%", "+100%"]
    assert len(ax.lines) == 2
    assert ax.lines[0].get_ydata()[0] == 1 and ax.lines[1].get_ydata()[0] == 2


def test_trend_labels_single_reference(mf):
    order = ["Original", "GR-seq", "Time", "P-seq"]
    _, ax = _trend(mf, order, [[9, 2, 3, float("nan")], [9, 2, 1, float("nan")]])
    assert [t.get_text() for t in ax._trend_labels] == ["+0%"]
    assert len(ax.lines) == 1 and ax.lines[0].get_ydata()[0] == 2


def test_labels_stay_inside_axes_and_ticks_unchanged(mf):
    fig, ax = _trend(mf, ["GR-seq", "Time"], [[1.99, 2.0]])
    ticks = ax.get_yticks().copy()
    mf._fit_trend_labels(fig)
    inside = ax.transAxes.inverted()
    assert all(t.get_window_extent().transformed(inside).y1 <= 0.98 for t in ax._trend_labels)
    assert ax.get_ylim()[1] > 2
    assert list(ax.get_yticks()) == [t for t in ticks if t <= 2]


def test_grid_renders(mf, tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "OUT_DIR", str(tmp_path))
    mf._save_group(mf.kinetics_group(mf.GRID2_DATASETS[2:]))
    assert (tmp_path / "ic50_grid.png").exists()


def test_legend_uses_paper_names(mf):
    labels = [
        p.get_label()
        for p in mf._legend_patches(["EITLEM", "HitEC", "Structure-Free DTA", "BIND"])
    ]
    assert labels == ["EITLEM-Kinetics", "HIT-EC", "Structure-Free DTA", "BIND"]
    assert "EITLEM" in mf.METHOD_COLORS  # colour lookup keeps the CSV name


def test_grid_methods_exist_in_csv(mf):
    import pandas as pd

    grids = [(key, ms) for _, key, ms in mf.GRID1_DATASETS + mf.GRID2_DATASETS]
    grids += [("ec_classification", mf.EC_METHODS), ("retrieval", mf.RET_METHODS)]
    grids += [(key, mf.RXNPRED_METHODS) for _, key in mf.RXNPRED_ROWS]
    for key, methods in grids:
        assert set(methods) <= set(pd.read_csv(mf.CSV[key])["Method"]), key
