"""
figures.py

Publication-quality (matplotlib) figure builders for single-cycle kinetic
fits and multi-cycle comparisons, plus a helper to render figures to bytes
for Streamlit download buttons.
"""

from __future__ import annotations

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinetic_models import MODELS, evaluate_fit_result_curve

PUB_RC = {
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.dpi": 130,
}


def _style():
    plt.rcParams.update(PUB_RC)


def fig_to_bytes(fig, fmt="png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_cycle_fit(cycle_result, fit_result, model_label: str,
                   model_key1: str | None = None, model_key2: str | None = None):
    """Data + two-stage fit overlay for one cycle. If model_key1 is given,
    the fitted curve is reconstructed at the cycle's full time resolution
    (fitting may have run on a downsampled subset for speed); otherwise the
    fit_result's own (possibly downsampled) X_pred/t grid is used."""
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    t = cycle_result.t_rel
    X = cycle_result.X

    ax.scatter(t, X, s=8, facecolors="none", edgecolors="#333333", linewidths=0.6,
              label="Experimental data", zorder=2)

    if fit_result is not None and fit_result.success:
        if model_key1 is not None:
            X_pred_full = evaluate_fit_result_curve(t, fit_result, model_key1, model_key2)
            ax.plot(t, X_pred_full, color="#C1272D", linewidth=2.0,
                    label="Two-stage fit", zorder=3)
        else:
            ax.plot(t, fit_result.X_pred, color="#C1272D", linewidth=2.0,
                    label="Two-stage fit", zorder=3)
        if fit_result.breakpoint_t is not None:
            ax.axvline(fit_result.breakpoint_t, color="#1B6CA8", linestyle="--",
                       linewidth=1.3, zorder=1,
                       label=f"Transition, t = {fit_result.breakpoint_t:.2f} min")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Fractional conversion, X (–)")
    title = f"{cycle_result.label} — {model_label}"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(bottom=min(0, X.min() - 0.02))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cycle_comparison_curves(cycle_results: list, title: str = "Conversion curves"):
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cmap = plt.get_cmap("viridis")
    n = max(len(cycle_results), 1)
    for i, cr in enumerate(cycle_results):
        color = cmap(i / max(n - 1, 1))
        ax.plot(cr.t_rel, cr.X, color=color, linewidth=1.8, label=cr.label)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Fractional conversion, X (–)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=1)
    fig.tight_layout()
    return fig


def plot_capacity_bar(labels, capacities, cycle_indices=None, title="CO₂ capture capacity per cycle"):
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(labels))
    ax.bar(x, capacities, color="#3B6E8F", edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Capacity (mmol CO₂ g⁻¹)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_deactivation_fit(cycle_numbers, capacities, fitted_curve=None, params=None,
                          title="Cycle-to-cycle capacity decay"):
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(cycle_numbers, capacities, s=30, color="#333333", zorder=3, label="Data")
    if fitted_curve is not None:
        ax.plot(fitted_curve[0], fitted_curve[1], color="#C1272D", linewidth=2.0,
               zorder=2, label="Grasa–Abanades fit")
    ax.set_xlabel("Cycle number, N")
    ax.set_ylabel("Capacity (mmol CO₂ g⁻¹)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig
