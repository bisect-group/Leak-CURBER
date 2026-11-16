"""Regenerate the result figures from the CSV tables in figures/data."""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from tueplots import bundles


# ─── TYPOGRAPHY ───────────────────────────────────────────────────────────────
FONT_FAMILY = "sans-serif"

FONT_SIZE = 14
FONT_SIZE_BASE = FONT_SIZE
FONT_SIZE_TITLE = FONT_SIZE
FONT_SIZE_TICK = FONT_SIZE
FONT_SIZE_LABEL = FONT_SIZE
FONT_SIZE_LEGEND = FONT_SIZE
FONT_SIZE_ANNOTATION = FONT_SIZE

# ─── TREND LABELS (% change vs reference split) ───────────────────────────────
FONT_SIZE_TREND = 10
DELTA_BOLD = False

TITLE_BOLD = False
TITLE_ITALIC = False
TITLE_UNDERLINE = False
LABEL_BOLD = False
LABEL_ITALIC = False
ANNOTATION_BOLD = False
ANNOTATION_ITALIC = False
TICK_BOLD = False

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
METHOD_COLORS = {
    "KcatNet": "#e41a1c",
    "EITLEM": "#377eb8",
    "CataPro": "#4daf4a",
    "DEKP": "#ff7f00",
    "eMOSAIC-T": "#984ea3",
    "BIND": "#a65628",
    "BACPI": "#f781bf",
    "Structure-Free DTA": "#1b9e77",
    "CLEAN": "#e41a1c",
    "GraphEC": "#377eb8",
    "HitEC": "#4daf4a",
    "CREEP": "#ff7f00",
    "CLIPZyme": "#984ea3",
    "Horizyn": "#a65628",
    "Root-aligned SMILES": "#d95f02",
    "MEGAN": "#7570b3",
    "Chemformer": "#e7298a",
}

# ─── BAR STYLE ────────────────────────────────────────────────────────────────
BAR_WIDTH = 0.22
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.4
ERROR_CAP_SIZE = 2
ERROR_LINE_WIDTH = 1

# ─── GRID ─────────────────────────────────────────────────────────────────────
GRID_COLOR = "#e6e6e6"
GRID_LINE_WIDTH = 0.6

# ─── AXIS LIMITS & TICKS ──────────────────────────────────────────────────────
RMSE_YLIM = (0, 2.0)
RMSE_YTICK_STEP = 0.25
PCC_YLIM = (0, 1.0)
PCC_YTICK_STEP = 0.1
ACC_YLIM = (0, 100)
ACC_YTICK_STEP = 10

# ─── TICK ROTATION ────────────────────────────────────────────────────────────
TICK_ROTATION = 30
TICK_HA = "right"

# ─── LEGEND ───────────────────────────────────────────────────────────────────
LEGEND_FRAME = False
LEGEND_NCOL_OVERRIDE = None
LEGEND_ANCHOR = (0.5, 1.04)

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
LAYOUT_ENGINE = "tight"
# top=0.88 reserves ~12% of canvas height above the subplots for the legend
# and out-of-axes annotations (trend labels). Increase if anything clips.
TIGHT_RECT = [0, 0, 1, 0.88]
TIGHT_PAD = 1
TIGHT_H_PAD = 2.4
TIGHT_W_PAD = 1.0

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT_DIR = str(HERE / "results_figures")
SAVE_PDF = False
SAVE_PNG = True
PNG_DPI = 300
SAVE_TRANSPARENT_BACKGROUND = True
# False → save the exact canvas (identical dimensions across same-type figures).
# True  → crop to content bounding box (convenient but sizes vary per figure).
SAVE_BBOX_TIGHT = True

# ─── DATA FILES ───────────────────────────────────────────────────────────────
DATA_DIR = str(HERE / "data")
CSV = {
    "kcat": os.path.join(DATA_DIR, "kcat.csv"),
    "km": os.path.join(DATA_DIR, "km.csv"),
    "ki": os.path.join(DATA_DIR, "ki.csv"),
    "kd": os.path.join(DATA_DIR, "kd.csv"),
    "ec50": os.path.join(DATA_DIR, "ec50.csv"),
    "ic50": os.path.join(DATA_DIR, "ic50.csv"),
    "ec_classification": os.path.join(DATA_DIR, "ec_classification.csv"),
    "retrieval": os.path.join(DATA_DIR, "retrieval.csv"),
    "retrosynthesis": os.path.join(DATA_DIR, "retrosynthesis.csv"),
    "forward_synthesis": os.path.join(DATA_DIR, "forward_synthesis.csv"),
}

# ─── FIGURE DATASETS ──────────────────────────────────────────────────────────
SPLITS_ORDER_KINETICS = [
    "Original",
    "GR-SMILES",
    "Mol-2D",
    "Mol-3D",
    "GR-seq",
    "Time",
    "P-seq",
    "P-struct",
]

GRID1_DATASETS = [
    (r"$k_{cat}$", "kcat", ["KcatNet", "EITLEM", "CataPro"]),
    (r"$K_M$", "km", ["EITLEM", "CataPro", "DEKP"]),
    (r"$K_i$", "ki", ["eMOSAIC-T", "BIND", "BACPI"]),
]

GRID2_DATASETS = [
    (r"$K_d$", "kd", ["BIND", "BACPI", "Structure-Free DTA"]),
    (r"$EC_{50}$", "ec50", ["BIND", "BACPI"]),
    (r"$IC_{50}$", "ic50", ["BIND", "BACPI"]),
]

EC_SPLITS_ORDER = ["Original", "GR-seq", "Time", "P-seq", "P-struct"]
EC_METHODS = ["CLEAN", "GraphEC", "HitEC"]

RET_SPLITS_ORDER = ["GR-rxnsmi", "Time", "DRFP", "P-seq", "P-struct"]
RET_METHODS = ["CREEP", "CLIPZyme", "Horizyn"]

LEVEL_PANELS = [
    ("L1", "Level 1 (X.-.-.-) (↑)"),
    ("L2", "Level 2 (X.X.-.-) (↑)"),
    ("L3", "Level 3 (X.X.X.-) (↑)"),
    ("L4", "Level 4 (X.X.X.X) (↑)"),
]

RXNPRED_SPLITS_ORDER = ["Original", "GR-rxnsmi", "Time", "DRFP"]
RXNPRED_METHODS = ["Root-aligned SMILES", "MEGAN", "Chemformer"]
RXNPRED_ROWS = [
    ("Retrosynthesis", "retrosynthesis"),
    ("Forward synthesis", "forward_synthesis"),
]
RXNPRED_COLS = [
    ("Top1", "Top-1 Acc. (%) (↑)"),
    ("Top10", "Top-10 Acc. (%) (↑)"),
]
RXNPRED_YLIM = (0, 100)
RXNPRED_YTICK_STEP = 10


# Apply rcParams from config — run automatically after CONFIG.
# Note: NeurIPS bundle sets text.usetex=True and font.family=serif.
# We disable usetex when the user picks a non-serif font, because LaTeX
# defaults to Computer Modern (serif) regardless of matplotlib's font.family.
plt.rcParams.update(bundles.neurips2024())


def _typography_overrides():
    """Return the dict of rcParams that mirror the CONFIG typography section."""
    return {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        # Only keep LaTeX rendering when the user actually wants serif.
        "text.usetex": FONT_FAMILY == "serif",
    }


plt.rcParams.update(_typography_overrides())


def _fw(bold, italic):
    """Return (fontweight, fontstyle) tuple."""
    return ("bold" if bold else "normal"), ("italic" if italic else "normal")


TITLE_FW, TITLE_FS = _fw(TITLE_BOLD, TITLE_ITALIC)
LABEL_FW, LABEL_FS = _fw(LABEL_BOLD, LABEL_ITALIC)
ANNOT_FW, ANNOT_FS = _fw(ANNOTATION_BOLD, ANNOTATION_ITALIC)
TICK_FW = "bold" if TICK_BOLD else "normal"
DELTA_FW = "bold" if DELTA_BOLD else "normal"

plt.rcParams["legend.columnspacing"] = 0.8
plt.rcParams["legend.handletextpad"] = 0.4


# Load all CSVs — empty cells become NaN automatically
dfs = {key: pd.read_csv(path) for key, path in CSV.items()}


def _apply_bundle(nrows, ncols):
    plt.rcParams.update(bundles.neurips2024(nrows=nrows, ncols=ncols))
    plt.rcParams.update(_typography_overrides())


def _set_tick_weight(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight(TICK_FW)


def _escape_for_latex(s):
    if not plt.rcParams.get("text.usetex", False):
        return s
    if "$" in s:
        return s
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _apply_title(ax, text):
    ax.set_title(" " * len(text), fontsize=FONT_SIZE_TITLE, pad=4, alpha=0)
    ax._deferred_title = text


def _finalize_titles(fig, pad=0.0):
    for ax in fig.axes:
        if not hasattr(ax, "_deferred_title"):
            continue
        text = _escape_for_latex(ax._deferred_title)
        pos = ax.get_position()
        x = (pos.x0 + pos.x1) / 2
        y = pos.y1 + pad
        fig.text(
            x,
            y,
            text,
            ha="center",
            va="bottom",
            fontweight=TITLE_FW,
            fontstyle=TITLE_FS,
            fontsize=FONT_SIZE_TITLE,
            transform=fig.transFigure,
            clip_on=False,
        )
        if TITLE_UNDERLINE:
            fig.text(
                x,
                y,
                r"$\underline{" + text.replace(" ", r"\ ") + "}$",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TITLE,
                transform=fig.transFigure,
                clip_on=False,
            )


def _make_fig(nrows, ncols, sharex=True, **kwargs):
    use_cl = LAYOUT_ENGINE == "constrained"
    return plt.subplots(nrows, ncols, sharex=sharex, constrained_layout=use_cl, **kwargs)


def _apply_layout(fig):
    if LAYOUT_ENGINE == "tight":
        tl_kwargs = {"rect": TIGHT_RECT, "pad": TIGHT_PAD}
        if TIGHT_H_PAD is not None:
            tl_kwargs["h_pad"] = TIGHT_H_PAD
        if TIGHT_W_PAD is not None:
            tl_kwargs["w_pad"] = TIGHT_W_PAD
        fig.tight_layout(**tl_kwargs)


def _add_subplot_annotation(ax, text, x=0.02, y=0.97, va="top", ha="left", bbox=None):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        fontweight=ANNOT_FW,
        fontstyle=ANNOT_FS,
        fontsize=FONT_SIZE_ANNOTATION,
        bbox=bbox,
    )


def _format_split_labels(splits_order):
    return [label.replace("smiles", "SMILES") for label in splits_order]


def _add_trend_labels(ax, x, method_means, method_stds, splits_order):
    means = np.asarray(method_means, dtype=float)
    avg = np.nanmean(means, axis=0)
    top = np.nanmax(means + np.asarray(method_stds, dtype=float), axis=0)
    half = 0.5 * BAR_WIDTH * len(means) + 0.02
    lo, hi = ax.get_ylim()
    pad = 0.015 * (hi - lo)
    idx = {s: i for i, s in enumerate(splits_order)}
    if "GR-SMILES" in idx:
        groups = [
            ("GR-SMILES", ["Mol-2D", "Mol-3D"], "Mol-3D"),
            ("GR-seq", ["Time", "P-seq", "P-struct"], "P-struct"),
        ]
    else:
        valid = [s for s in splits_order if s != "Original" and np.isfinite(avg[idx[s]])]
        groups = [(valid[0], valid[1:], splits_order[-1])]
    ax._trend_labels = []
    for ref_name, targets, last in groups:
        ref = idx[ref_name]
        ax.plot(
            [x[ref] - half, x[idx[last]] + half],
            [avg[ref]] * 2,
            color="black",
            ls="--",
            lw=1.2,
            alpha=0.8,
            zorder=6,
        )
        for name in targets:
            i = idx[name]
            pct = (avg[i] - avg[ref]) / abs(avg[ref]) * 100
            label = ax.text(
                x[i],
                max(top[i], avg[ref]) + pad,
                f"{pct:+.0f}%",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_TREND,
                fontweight=DELTA_FW,
                zorder=8,
            )
            ax._trend_labels.append(label)


def _fit_trend_labels(fig):
    """Raise the upper y-limit just enough that no trend label leaves the axes."""
    fig.canvas.draw()
    for ax in fig.axes:
        labels = getattr(ax, "_trend_labels", [])
        if not labels:
            continue
        lo, hi = ax.get_ylim()
        ticks = ax.get_yticks()
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        for _ in range(5):
            top = max(
                t.get_window_extent().transformed(ax.transAxes.inverted()).y1 for t in labels
            )
            if top <= 0.98:
                break
            hi = lo + (hi - lo) * top / 0.97
            ax.set_ylim(lo, hi)
            fig.canvas.draw()
        ax.set_yticks(ticks)
        ax.set_ylim(lo, hi)
    fig.canvas.draw()


def _set_y_label(ax, text):
    ax.set_ylabel(text, fontweight=LABEL_FW, fontstyle=LABEL_FS, fontsize=FONT_SIZE_LABEL)


def plot_rmse_pcc_bars(ax, df, methods, metric, splits_order, trend_arrow=False):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    x = np.arange(len(splits_order))
    n = len(methods)
    method_means = []
    method_stds = []

    for idx, method in enumerate(methods):
        m = df[df["Method"] == method].set_index("Split").reindex(splits_order)
        means = m[mean_col].to_numpy(dtype=float)
        stds = np.where(
            np.isnan(m[std_col].to_numpy(dtype=float)),
            0.0,
            m[std_col].to_numpy(dtype=float),
        )
        method_means.append(means)
        method_stds.append(stds)
        offset = (idx - (n - 1) / 2) * BAR_WIDTH
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            color=METHOD_COLORS[method],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            yerr=stds,
            error_kw={
                "capsize": ERROR_CAP_SIZE,
                "elinewidth": ERROR_LINE_WIDTH,
                "zorder": 5,
            },
            label=method,
        )

    if metric == "PCC":
        ax.set_ylim(*PCC_YLIM)
        ax.set_yticks(np.arange(PCC_YLIM[0], PCC_YLIM[1] + 1e-9, PCC_YTICK_STEP))
    else:
        ax.set_ylim(*RMSE_YLIM)
        ax.set_yticks(np.arange(RMSE_YLIM[0], RMSE_YLIM[1] + 1e-9, RMSE_YTICK_STEP))

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(_format_split_labels(splits_order), rotation=TICK_ROTATION, ha=TICK_HA)
    ax.margins(x=0.08)
    if trend_arrow:
        _add_trend_labels(ax, x, method_means, method_stds, splits_order)
    sns.despine(ax=ax)
    _set_tick_weight(ax)


def plot_accuracy_bars(ax, df, methods, metric, splits_order, trend_arrow=False):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    x = np.arange(len(splits_order))
    n = len(methods)
    method_means = []
    method_stds = []

    for idx, method in enumerate(methods):
        m = df[df["Method"] == method].set_index("Split").reindex(splits_order)
        means = m[mean_col].to_numpy(dtype=float)
        stds = np.where(
            np.isnan(m[std_col].to_numpy(dtype=float)),
            0.0,
            m[std_col].to_numpy(dtype=float),
        )
        method_means.append(means)
        method_stds.append(stds)
        offset = (idx - (n - 1) / 2) * BAR_WIDTH
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            color=METHOD_COLORS[method],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            yerr=stds,
            error_kw={
                "capsize": ERROR_CAP_SIZE,
                "elinewidth": ERROR_LINE_WIDTH,
                "zorder": 5,
            },
            label=method,
        )

    ax.set_ylim(*ACC_YLIM)
    ax.set_yticks(np.arange(ACC_YLIM[0], ACC_YLIM[1] + 1e-9, ACC_YTICK_STEP))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(_format_split_labels(splits_order), rotation=TICK_ROTATION, ha=TICK_HA)
    ax.margins(x=0.08)
    if trend_arrow:
        _add_trend_labels(ax, x, method_means, method_stds, splits_order)
    sns.despine(ax=ax)
    _set_tick_weight(ax)


# Legend text only; data lookup and colours keep the CSV names.
LEGEND_LABELS = {
    "EITLEM": "EITLEM-Kinetics",
    "HitEC": "HIT-EC",
}


def _legend_patches(methods):
    return [
        mpatches.Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=LEGEND_LABELS.get(m, m),
        )
        for m in methods
    ]


def _add_fig_legend(fig, methods):
    ncol = LEGEND_NCOL_OVERRIDE if LEGEND_NCOL_OVERRIDE is not None else len(methods)
    fig.legend(
        handles=_legend_patches(methods),
        loc="upper center",
        ncol=ncol,
        frameon=LEGEND_FRAME,
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=FONT_SIZE_LEGEND,
    )


def _add_row_legend(fig, row_axes, methods):
    axes = np.ravel(row_axes)
    row_top = max(ax.get_position().y1 for ax in axes)
    ncol = LEGEND_NCOL_OVERRIDE if LEGEND_NCOL_OVERRIDE is not None else len(methods)
    fig.legend(
        handles=_legend_patches(methods),
        loc="lower center",
        ncol=ncol,
        frameon=LEGEND_FRAME,
        bbox_to_anchor=(0.5, row_top + 0.008),
        bbox_transform=fig.transFigure,
        fontsize=FONT_SIZE_LEGEND,
    )


def _common_axes_positions(group):
    axis_groups = [np.ravel(item["axes"]) for item in group]
    n_axes = len(axis_groups[0])
    if any(len(axes) != n_axes for axes in axis_groups):
        raise ValueError("Grouped figures must have the same number of axes.")

    common = []
    for axis_idx in range(n_axes):
        boxes = [axes[axis_idx].get_position() for axes in axis_groups]
        x0 = max(box.x0 for box in boxes)
        y0 = max(box.y0 for box in boxes)
        x1 = min(box.x1 for box in boxes)
        y1 = min(box.y1 for box in boxes)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Grouped figure axes do not overlap enough to align.")
        common.append([x0, y0, x1 - x0, y1 - y0])

    for axes in axis_groups:
        for ax, pos in zip(axes, common):
            ax.set_position(pos)


def _shared_tight_bbox(figs):
    bboxes = []
    for fig in figs:
        fig.canvas.draw()
        bboxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
    return mtransforms.Bbox.from_extents(
        min(box.x0 for box in bboxes),
        min(box.y0 for box in bboxes),
        max(box.x1 for box in bboxes),
        max(box.y1 for box in bboxes),
    )


def _save_group(group):
    if not group:
        return
    _common_axes_positions(group)
    for item in group:
        _add_row_legend(item["fig"], item["axes"], item["methods"])
    bbox_inches = _shared_tight_bbox([item["fig"] for item in group]) if SAVE_BBOX_TIGHT else None
    for item in group:
        _save(item["fig"], item["base"], bbox_inches=bbox_inches)


def _save(fig, base, bbox_inches=None):
    _fit_trend_labels(fig)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, base)
    save_kw = {"transparent": SAVE_TRANSPARENT_BACKGROUND}
    if bbox_inches is not None:
        save_kw["bbox_inches"] = bbox_inches
    elif SAVE_BBOX_TIGHT:
        save_kw["bbox_inches"] = "tight"
    if SAVE_PDF:
        fig.savefig(f"{path}.pdf", **save_kw)
    if SAVE_PNG:
        fig.savefig(f"{path}.png", dpi=PNG_DPI, **save_kw)
    print(f"Saved → {path}.pdf / .png")


def kinetics_group(datasets):
    group = []
    for label, csv_key, methods in datasets:
        _apply_bundle(nrows=2, ncols=1)
        df = dfs[csv_key]
        fig, axes = _make_fig(2, 1)

        for row, metric in enumerate(["RMSE", "PCC"]):
            ax = axes[row]
            plot_rmse_pcc_bars(
                ax,
                df,
                methods,
                metric,
                SPLITS_ORDER_KINETICS,
                trend_arrow=True,
            )
            _set_y_label(ax, f"{metric} ({'↓' if metric == 'RMSE' else '↑'})")
            _add_subplot_annotation(ax, label)

        _apply_layout(fig)
        group.append({"fig": fig, "axes": axes, "base": f"{csv_key}_grid", "methods": methods})
    return group


def ec_function_group():
    ec_function_group = []
    _apply_bundle(nrows=2, ncols=2)
    w, h = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (w * 1.5, h * 2)

    ec_function_group = []
    fig3, axes3 = _make_fig(2, 2)
    axes3_flat = np.ravel(axes3)
    df_ec = dfs["ec_classification"]

    for idx, (metric, title) in enumerate(LEVEL_PANELS):
        ax = axes3_flat[idx]
        plot_accuracy_bars(
            ax,
            df_ec,
            EC_METHODS,
            metric,
            EC_SPLITS_ORDER,
            trend_arrow=True,
        )
        _set_y_label(ax, title)

    _apply_layout(fig3)
    ec_function_group.append(
        {
            "fig": fig3,
            "axes": axes3_flat,
            "base": "ec_classification_grid",
            "methods": EC_METHODS,
        }
    )
    _apply_bundle(nrows=2, ncols=2)
    w, h = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (w * 1.5, h * 2)

    fig4, axes4 = _make_fig(2, 2)
    axes4_flat = np.ravel(axes4)
    df_ret = dfs["retrieval"]

    for idx, (metric, title) in enumerate(LEVEL_PANELS):
        ax = axes4_flat[idx]
        plot_accuracy_bars(
            ax,
            df_ret,
            RET_METHODS,
            metric,
            RET_SPLITS_ORDER,
            trend_arrow=True,
        )
        _set_y_label(ax, title)

    _apply_layout(fig4)
    ec_function_group.append(
        {"fig": fig4, "axes": axes4_flat, "base": "retrieval_grid", "methods": RET_METHODS}
    )
    return ec_function_group


def rxnpred_group():
    rxnpred_group = []
    for panel_label, csv_key in RXNPRED_ROWS:
        _apply_bundle(nrows=1, ncols=2)
        w, h = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (w * 1.5, h * 2.25)

        df = dfs[csv_key]
        fig, axes = _make_fig(1, 2)

        x = np.arange(len(RXNPRED_SPLITS_ORDER))
        n = len(RXNPRED_METHODS)

        for idx, (metric_prefix, col_title) in enumerate(RXNPRED_COLS):
            ax = axes[idx]
            method_means = []
            method_stds = []
            for m_idx, method in enumerate(RXNPRED_METHODS):
                m = df[df["Method"] == method].set_index("Split").reindex(RXNPRED_SPLITS_ORDER)
                means = m[f"{metric_prefix}_mean"].to_numpy(dtype=float)
                stds = np.where(
                    np.isnan(m[f"{metric_prefix}_std"].to_numpy(dtype=float)),
                    0.0,
                    m[f"{metric_prefix}_std"].to_numpy(dtype=float),
                )
                method_means.append(means)
                method_stds.append(stds)
                offset = (m_idx - (n - 1) / 2) * BAR_WIDTH
                ax.bar(
                    x + offset,
                    means,
                    width=BAR_WIDTH,
                    color=METHOD_COLORS[method],
                    edgecolor=BAR_EDGE_COLOR,
                    linewidth=BAR_EDGE_WIDTH,
                    yerr=stds,
                    error_kw={
                        "capsize": ERROR_CAP_SIZE,
                        "elinewidth": ERROR_LINE_WIDTH,
                        "zorder": 5,
                    },
                    label=method,
                )

            ax.set_ylim(*RXNPRED_YLIM)
            ax.set_yticks(np.arange(RXNPRED_YLIM[0], RXNPRED_YLIM[1] + 1e-9, RXNPRED_YTICK_STEP))
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
            ax.set_xticks(x)
            ax.set_xticklabels(_format_split_labels(RXNPRED_SPLITS_ORDER), rotation=30, ha="right")
            ax.margins(x=0.08)
            _add_trend_labels(ax, x, method_means, method_stds, RXNPRED_SPLITS_ORDER)
            sns.despine(ax=ax)
            _set_tick_weight(ax)
            _set_y_label(ax, col_title)

        _apply_layout(fig)
        rxnpred_group.append(
            {
                "fig": fig,
                "axes": axes,
                "base": f"{csv_key}_grid",
                "methods": RXNPRED_METHODS,
            }
        )
    return rxnpred_group


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=OUT_DIR)
    OUT_DIR = parser.parse_args().out_dir
    _save_group(kinetics_group(GRID1_DATASETS))
    _save_group(kinetics_group(GRID2_DATASETS))
    _save_group(ec_function_group())
    _save_group(rxnpred_group())
    plt.close("all")


if __name__ == "__main__":
    main()
