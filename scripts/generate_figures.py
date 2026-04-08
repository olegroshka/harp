#!/usr/bin/env python
"""
Generate paper figures (v3) for the heterogeneity-aware forecasting paper.

Figures:
  2. Per-window R^2 comparison: G1 (global) vs M2 (mixture PCA+ridge)
  3. Placebo distribution histogram
  4. Geodesic distance: global vs local blocks
  5. Per-block R^2 by architecture (G0, G1, M2)

Figure 1 (architecture diagram) is TikZ in LaTeX -- not generated here.

Saves PDF + PNG to docs/smim/paper/figures/.

Usage::
    uv run python scripts/smim/paper_figures_v3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS = PROJECT_ROOT / "results" / "metrics"

# ── Colour palette ──────────────────────────────────────────────────────────
# Blues for global models, reds/darks for mixture/local
BLUE_GLOBAL = "#4878A8"     # G1 global
BLUE_POOLED = "#9BBAD8"     # G0 pooled (lighter blue)
RED_MIXTURE = "#B33030"     # M2 mixture
GREY_REF    = "#888888"     # reference lines
GREY_LIGHT  = "#BBBBBB"     # secondary elements
GREY_BAR    = "#999999"     # placebo bars
BLACK       = "#222222"

# ── Matplotlib RC ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
})


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.pdf / {name}.png")


def _despine(ax: plt.Axes, left: bool = True) -> None:
    """Remove top and right spines; optionally keep left."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Per-window R^2 -- G1 vs M2 (connected dot plot)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_per_window_r2() -> None:
    df = pd.read_parquet(METRICS / "iter6_4b.parquet")

    g1 = df[df["architecture"] == "G1"].sort_values("year")
    m2 = df[df["architecture"] == "M2"].sort_values("year")

    years = g1["year"].values.astype(int)
    r2_g1 = g1["full_r2"].values
    r2_m2 = m2["full_r2"].values

    # Reference baselines (G0 pooled+FE mean, approximate AR(1))
    ar1_ref = 0.610
    pooled_fe_ref = 0.591

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    x = np.arange(len(years))

    # Vertical connectors (thin grey lines connecting G1 to M2)
    for i in range(len(years)):
        ax.plot([x[i], x[i]], [r2_g1[i], r2_m2[i]],
                color=GREY_LIGHT, linewidth=1.2, zorder=1)

    # Dots
    ax.scatter(x, r2_g1, color=BLUE_GLOBAL, s=55, zorder=3,
               label=f"G1 global (mean {r2_g1.mean():.3f})", edgecolors="white",
               linewidths=0.5)
    ax.scatter(x, r2_m2, color=RED_MIXTURE, s=55, zorder=3, marker="D",
               label=f"M2 mixture (mean {r2_m2.mean():.3f})", edgecolors="white",
               linewidths=0.5)

    # Reference lines
    ax.axhline(ar1_ref, color=GREY_REF, linestyle="--", linewidth=0.8,
               zorder=0, alpha=0.7)
    ax.text(len(years) - 0.5, ar1_ref + 0.004, f"AR(1) = {ar1_ref:.3f}",
            fontsize=8, color=GREY_REF, ha="right", va="bottom")

    ax.axhline(pooled_fe_ref, color=GREY_REF, linestyle=":", linewidth=0.8,
               zorder=0, alpha=0.7)
    ax.text(len(years) - 0.5, pooled_fe_ref - 0.004, f"Pooled+FE = {pooled_fe_ref:.3f}",
            fontsize=8, color=GREY_REF, ha="right", va="top")

    # Delta annotations above each M2 point
    for i in range(len(years)):
        delta = r2_m2[i] - r2_g1[i]
        ax.text(x[i], r2_m2[i] + 0.012, f"+{delta:.3f}",
                ha="center", fontsize=7, color=RED_MIXTURE, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_xlabel("Test year")
    ax.set_ylim(0.48, 0.78)
    ax.legend(loc="lower right", fontsize=8)
    _despine(ax)

    fig.tight_layout()
    _save(fig, "fig2_per_window_r2")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Placebo distribution
# ═══════════════════════════════════════════════════════════════════════════

def fig3_placebo() -> None:
    df = pd.read_parquet(METRICS / "iter6_4b_placebo.parquet")

    real_row = df[df["type"] == "real"]
    placebo = df[df["type"] == "placebo"]

    real_delta = real_row["mean_delta"].values[0]
    placebo_deltas = placebo["mean_delta"].values

    # Statistics
    z_score = (real_delta - placebo_deltas.mean()) / placebo_deltas.std()

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Histogram of placebo deltas
    n_bins = 35
    _n, _bins, _patches = ax.hist(
        placebo_deltas, bins=n_bins, color=GREY_BAR, edgecolor="white",
        linewidth=0.8, alpha=0.85, zorder=2, label="Placebo partitions ($n=1{,}000$)",
    )

    # Real value as vertical dashed line
    ax.axvline(real_delta, color=RED_MIXTURE, linestyle="--", linewidth=2.0,
               zorder=3, label=f"Real blocks ($\\Delta = +{real_delta:.3f}$)")

    # Annotation -- position relative to histogram peak
    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"$z = {z_score:.2f}$\n$p < 0.001$",
        xy=(real_delta, ymax * 0.15),
        xytext=(real_delta - 0.018, ymax * 0.72),
        fontsize=10,
        color=RED_MIXTURE,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=RED_MIXTURE, lw=1.0),
    )

    ax.set_xlabel("Mean $\\Delta R^2$ (M2 $-$ G1)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=8)
    _despine(ax)

    fig.tight_layout()
    _save(fig, "fig3_placebo")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Geodesic distance -- global vs local blocks
# ═══════════════════════════════════════════════════════════════════════════

def fig4_geodesic() -> None:
    df = pd.read_parquet(METRICS / "iter6_4_gate_b_diagnostics.parquet")

    # Sort by geodesic distance ascending (smallest at bottom for horizontal bars)
    df_sorted = df.sort_values("geodesic_deg", ascending=True).copy()

    # Clean up block names for display
    name_map = {
        "GLOBAL": "GLOBAL ($N=93$)",
        "LAYER_firms": "Firms layer ($N=82$)",
        "LAYER_macro_inst": "Macro + institutional ($N=11$)",
        "MERGED_ind_energy": "Industrials + energy ($N=26$)",
        "MERGED_tech_health": "Technology + healthcare ($N=25$)",
        "SEC_diversified": "Diversified ($N=23$)",
        "SEC_energy": "Energy ($N=14$)",
        "SEC_financials": "Financials ($N=12$)",
        "SEC_healthcare": "Healthcare ($N=10$)",
        "SEC_industrials": "Industrials ($N=12$)",
        "SEC_technology": "Technology ($N=15$)",
    }
    labels = [name_map.get(b, b) for b in df_sorted["block"]]
    geodesic = df_sorted["geodesic_deg"].values
    blocks = df_sorted["block"].values
    is_global = blocks == "GLOBAL"

    # Colour: global in blue, high-rotation aggregates (>25 deg) in medium grey,
    # coherent local blocks (<20 deg) in red/accent
    colors = []
    for blk, val in zip(blocks, geodesic):
        if blk == "GLOBAL":
            colors.append(BLUE_GLOBAL)
        elif val > 25:
            colors.append(GREY_BAR)         # aggregate blocks that are ~global
        else:
            colors.append(RED_MIXTURE)      # coherent local blocks

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    y = np.arange(len(labels))

    bars = ax.barh(y, geodesic, color=colors, edgecolor="white", linewidth=0.6,
                   height=0.65, zorder=2)

    # Value labels at end of each bar
    for i, (val, blk) in enumerate(zip(geodesic, blocks)):
        fw = "bold" if blk == "GLOBAL" else "normal"
        c = BLUE_GLOBAL if blk == "GLOBAL" else BLACK
        ax.text(val + 0.4, i, f"{val:.1f}$^\\circ$",
                va="center", fontsize=8, color=c, fontweight=fw)

    # Bracket / annotation showing the gap
    # Find boundary between coherent (<20) and aggregate (>25) blocks
    coherent_top = max(i for i, v in enumerate(geodesic) if v < 20)
    aggregate_bot = min(i for i, v in enumerate(geodesic) if v > 25)
    if aggregate_bot > coherent_top + 1:
        mid_y = (coherent_top + aggregate_bot) / 2
        ax.axhline(mid_y, color=GREY_LIGHT, linestyle="--", linewidth=0.7,
                   zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Geodesic distance (degrees per quarter)")
    ax.set_xlim(0, 40)
    _despine(ax)

    fig.tight_layout()
    _save(fig, "fig4_geodesic")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Per-block R^2 by architecture (G0, G1, M2)
# ═══════════════════════════════════════════════════════════════════════════

def fig5_block_r2() -> None:
    df = pd.read_parquet(METRICS / "iter6_4b.parquet")

    block_cols = [
        "block_SEC_diversified",
        "block_LAYER_macro_inst",
        "block_MERGED_tech_health",
        "block_REMAINDER",
    ]
    block_labels = [
        "Diversified",
        "Macro +\ninstitutional",
        "Technology +\nhealthcare",
        "Remainder",
    ]

    archs = ["G0", "G1", "M2"]
    arch_labels = ["G0 (pooled)", "G1 (global)", "M2 (mixture)"]
    arch_colors = [BLUE_POOLED, BLUE_GLOBAL, RED_MIXTURE]

    # Compute mean R^2 per block per architecture
    means = {}
    for arch in archs:
        sub = df[df["architecture"] == arch]
        means[arch] = [sub[col].mean() for col in block_cols]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    n_blocks = len(block_cols)
    n_archs = len(archs)
    bar_width = 0.22
    x = np.arange(n_blocks)

    for j, (arch, label, color) in enumerate(zip(archs, arch_labels, arch_colors)):
        offset = (j - 1) * bar_width  # center the 3 bars
        vals = means[arch]
        bars = ax.bar(x + offset, vals, bar_width, color=color, edgecolor="white",
                      linewidth=0.6, label=label, zorder=2)

    # Highlight the diversified block where G1 < G0 (global hurts)
    g0_div = means["G0"][0]
    g1_div = means["G1"][0]
    # Position annotation to the right of the diversified group
    ax.annotate(
        "global\nhurts",
        xy=(0, g1_div),  # point at G1 bar top
        xytext=(0.7, 0.30),
        fontsize=8,
        color=BLUE_GLOBAL,
        fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=BLUE_GLOBAL, lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(block_labels, fontsize=9)
    ax.set_ylabel("Mean out-of-sample $R^2$")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0.25, 0.88)
    _despine(ax)

    # Light horizontal reference line at overall G0 mean
    g0_mean = df[df["architecture"] == "G0"]["full_r2"].mean()
    ax.axhline(g0_mean, color=GREY_LIGHT, linestyle=":", linewidth=0.7,
               zorder=0)

    fig.tight_layout()
    _save(fig, "fig5_block_r2")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Generating paper figures to {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    print()
    print("Figure 1: architecture diagram (TikZ in LaTeX -- skipped)")
    print()

    print("Figure 2: Per-window R^2 (G1 vs M2)")
    fig2_per_window_r2()

    print("Figure 3: Placebo distribution")
    fig3_placebo()

    # Figure 4 (geodesic) is not referenced in the paper LaTeX — skip.

    print("Figure 5: Per-block R^2 by architecture")
    fig5_block_r2()

    print()
    print("Done. All figures saved as PDF + PNG.")


if __name__ == "__main__":
    main()
