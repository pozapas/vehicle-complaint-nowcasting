"""Step 6  -  Figures 1-4, exactly per outline §5 panel specs.

All styling comes from figstyle.py; there are no local rcParam overrides here.
Every number plotted is read from a results/ file written by an earlier step.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from common import (AUDIT_DIR, D_CAP, FIG_DIR, HORIZONS, SEV_LABELS, T_CUTOFF, TAB_DIR,
                    p_to_str, str_to_p)

SEV_STREAMS = ("Y0", "Y1", "Y2", "Y3")
SHORT = {0: "no injury", 1: "crash/fire", 2: "injury", 3: "fatality"}


def R(name):
    return pd.read_csv(os.path.join(TAB_DIR, name))


def A(name):
    return pd.read_csv(os.path.join(AUDIT_DIR, name))


# ==================================================================== Figure 1
def figure1():
    """Method pipeline schematic: two clocks, the triangle, the workflow."""
    fig = plt.figure(figsize=(fs.W2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.34)

    # ---- (a) two-clock timeline of one complaint
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fs.panel_label(ax, "a", dx=-0.02, dy=0.99)
    ax.annotate("", xy=(9.6, 3.0), xytext=(0.3, 3.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.INK))
    ax.text(9.6, 2.2, "time", ha="right", fontsize=6.5, color=fs.MUTED)
    for x, ly, lab, col in ((1.3, 3.6, "incident\nFAILDATE", fs.SEV_COLORS[3]),
                            (4.5, 5.2, "consumer\ndecides", fs.MUTED),
                            (7.2, 3.6, "NHTSA receipt\nLDATE", fs.NOWCAST)):
        ax.plot([x], [3.0], "o", ms=4, color=col, zorder=3)
        ax.plot([x, x], [3.15, ly - 0.08], lw=0.4, color=col, alpha=0.5)
        ax.text(x, ly, lab, ha="center", va="bottom", fontsize=6.5, color=fs.INK)
    ax.annotate("", xy=(7.2, 2.35), xytext=(1.3, 2.35),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.6, color=fs.SEV_COLORS[2]))
    ax.text(4.25, 1.7, "reporting delay $d$", ha="center", fontsize=7,
            color=fs.SEV_COLORS[2])
    ax.plot([8.9, 8.9], [1.2, 6.0], ls=(0, (2, 1.6)), lw=0.7, color=fs.SEV_COLORS[3])
    ax.text(8.75, 6.2, "today $T$", fontsize=6.5, ha="right",
            color=fs.SEV_COLORS[3])
    ax.text(0.3, 8.6, "Two clocks per complaint", fontsize=7.5, fontweight="bold")
    ax.text(0.3, 7.5, "incidents already occurred but\nnot yet reported are invisible",
            fontsize=6.5, color=fs.MUTED)

    # ---- (b) the reporting triangle
    ax = fig.add_subplot(gs[1])
    fs.panel_label(ax, "b", dx=-0.16, dy=1.02)
    n = 9
    for i in range(n):
        for d in range(n):
            obs = (i + d) < n
            ax.add_patch(mpatches.Rectangle(
                (d, n - 1 - i), 1, 1,
                facecolor=fs.NOWCAST if obs else "#FFFFFF",
                alpha=0.75 if obs else 1.0,
                edgecolor="#FFFFFF" if obs else fs.SEV_COLORS[3],
                hatch=None if obs else "///", lw=0.4))
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xlabel("reporting delay $d$ (months)")
    ax.set_ylabel("incident cohort $t$")
    ax.set_xticks([0.5, n - 0.5]); ax.set_xticklabels(["0", "$D$"])
    ax.set_yticks([0.5, n - 0.5]); ax.set_yticklabels(["$T$", "older"])
    ax.tick_params(length=0)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.text(0.20, 0.52, "observed\n$t+d \\leq T$", transform=ax.transAxes,
            fontsize=6.5, color="#FFFFFF", ha="center", fontweight="bold")
    ax.text(0.76, 0.20, "not yet\nreported", transform=ax.transAxes, fontsize=6.5,
            color=fs.SEV_COLORS[3], ha="center", fontweight="bold",
            bbox=dict(facecolor="#FFFFFF", edgecolor="none", pad=1.0, alpha=0.85))

    # ---- (c) pipeline
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    fs.panel_label(ax, "c", dx=-0.02, dy=0.99)
    steps = ["complaint-level\nreconstruction", "delay estimation\n(mature cohorts)",
             "chain-ladder\nnowcast", "rolling-origin\nbacktest",
             "early-warning\napplication"]
    cols = ["#E8E8E8", fs.SEV_COLORS[1], fs.NOWCAST, fs.SEV_COLORS[2], fs.SEV_COLORS[3]]
    y = 8.6
    for k, (s, c) in enumerate(zip(steps, cols)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5, y - 1.15), 8.4, 1.15, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=c, edgecolor="none", alpha=0.9))
        ax.text(4.7, y - 0.57, s, ha="center", va="center", fontsize=6.4,
                color="#FFFFFF" if k >= 1 else fs.INK,
                fontweight="bold" if k == 2 else "normal")
        if k < len(steps) - 1:
            ax.annotate("", xy=(4.7, y - 1.55), xytext=(4.7, y - 1.2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.MUTED))
        y -= 1.72
    fs.save_fig(fig, "fig1")


def figure1_reporting_triangle_inset():
    """Standalone transparent reporting-triangle inset for the editable Fig. 1."""
    n = 9
    fig, ax = plt.subplots(figsize=(4.35, 2.85))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for i in range(n):
        for d in range(n):
            observed = (i + d) < n
            ax.add_patch(mpatches.Rectangle(
                (d, n - 1 - i), 1, 1,
                facecolor=fs.NOWCAST if observed else "#FDE9E5",
                edgecolor="#FFFFFF" if observed else fs.SEV_COLORS[3],
                hatch=None if observed else "////",
                linewidth=0.72,
                alpha=0.92 if observed else 0.72,
            ))

    # Frontier separating cells observable by the pseudo-real-time cutoff T*.
    ax.plot([0.98, n - 0.02], [-0.02, n - 0.02],
            color=fs.SEV_COLORS[3], lw=1.05, ls=(0, (3, 1.8)), zorder=4)
    ax.text(1.75, 6.0, "Observed\nreports", ha="center", va="center",
            color="#FFFFFF", fontsize=7.2, fontweight="bold", zorder=5)
    ax.text(6.35, 2.05, "Not yet\nreported", ha="center", va="center",
            color=fs.SEV_COLORS[3], fontsize=7.15, fontweight="bold", zorder=5)
    ax.text(6.65, 7.35, r"Censoring frontier: $t+d=T^*$",
            color=fs.SEV_COLORS[3], fontsize=6.2, rotation=34,
            ha="center", va="center", zorder=5,
            bbox=dict(facecolor="#FFFFFF", edgecolor="none", pad=0.4, alpha=0.78))

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([0.5, 3.5, 6.5, 8.5])
    ax.set_xticklabels(["0", "3", "6", "$D$"])
    ax.set_yticks([0.5, 4.5, 8.5])
    ax.set_yticklabels(["Recent", "", "Older"])
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Reporting Delay, $d$ (Months)", fontweight="bold", labelpad=3)
    ax.set_ylabel("Incident Cohort, $t$", fontweight="bold", labelpad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    out = os.path.join(FIG_DIR, "fig1_reporting_triangle_300dpi.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return out


# ==================================================================== Figure 2
def figure2():
    """Severity-specific reporting-delay atlas (mature cohorts only)."""
    ec = R("ecdf_by_severity.csv")
    hz = R("hazard_model.csv")
    dr = R("delay_drift.csv")

    fig, axes = plt.subplots(2, 2, figsize=(fs.W2, 4.72))
    fig.subplots_adjust(left=0.075, right=0.992, top=0.950, bottom=0.105,
                        hspace=0.42, wspace=0.42)

    # ---- (a) severity-stratified delay profiles
    ax = axes[0, 0]
    for s in range(4):
        g = ec[ec.severity == s].sort_values("d")
        median_month = g.loc[g["F"] >= 0.5, "d"].iloc[0]
        ax.plot(g["d"] + 1, g["F"], color=fs.SEV_COLORS[s], lw=1.35,
                ls=fs.SEV_LS[s], drawstyle="steps-post", label=SHORT[s], zorder=3)
        ax.plot(median_month + 1, 0.5, marker=fs.SEV_MARKER[s], ms=4.0,
                color=fs.SEV_COLORS[s], mec="white", mew=0.45, zorder=4)
    ax.axhline(0.5, color="#B7B7B7", lw=0.55, ls=(0, (2.0, 2.0)), zorder=1)
    ax.text(36.5, 0.515, "Median", ha="right", va="bottom", fontsize=5.55,
            color=fs.MUTED)
    ax.legend(fontsize=5.95, ncol=2, loc="lower right", borderaxespad=0.35,
              columnspacing=0.8, handlelength=1.65, handletextpad=0.35)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 7, 13, 25, 37])
    ax.set_xticklabels(["0", "1", "3", "6", "12", "24", "36"])
    ax.set_xlim(1, 37); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reporting Delay, $d$ (Months)", fontweight="bold")
    # empirical mature-cohort CDF, not the chain-ladder F_hat used for the correction
    ax.set_ylabel(r"Cumulative Share Reported, $\tilde{F}_s(d)$", fontweight="bold")
    fs.light_grid(ax)
    ax.set_title("(a) Severity-Stratified Reporting Delay", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.text(0.015, 0.055, "Markers identify median delay", transform=ax.transAxes,
            fontsize=5.45, color=fs.MUTED, ha="left", va="bottom")

    # ---- (b) severity effects on the reporting hazard
    ax = axes[0, 1]
    keep = hz[hz["term"].str.startswith(("severity_", "era_", "channel_"))].copy()
    pretty = {"severity_1": "crash/fire (vs none)", "severity_2": "injury (vs none)",
              "severity_3": "fatality (vs none)"}
    keep["lab"] = keep["term"].map(pretty)
    keep = keep[keep["lab"].notna()].iloc[::-1]
    ypos = np.arange(len(keep))
    ax.axvspan(0.60, 1.00, color=fs.SEV_COLORS[3], alpha=0.045, zorder=0)
    ax.axvspan(1.00, 1.40, color=fs.SEV_COLORS[1], alpha=0.050, zorder=0)
    for k, (_, r) in enumerate(keep.iterrows()):
        s = int(r["term"].split("_")[1])
        ax.errorbar(r["hr"], ypos[k],
                    xerr=[[r["hr"] - r["hr_lo95"]], [r["hr_hi95"] - r["hr"]]],
                    fmt=fs.SEV_MARKER[s], color=fs.SEV_COLORS[s], ms=4.9,
                    mec="white", mew=0.45, elinewidth=1.35, capsize=1.8,
                    capthick=0.7, zorder=3)
        label_x = r["hr_hi95"] + 0.025
        ax.text(label_x, ypos[k], f"{r['hr']:.2f}", va="center", fontsize=5.8,
                color=fs.SEV_COLORS[s], fontweight="bold")
    ax.axvline(1.0, color=fs.INK, lw=0.65, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(keep["lab"], fontsize=6.35)
    ax.tick_params(axis="y", pad=4)
    ax.set_xlabel("Reporting-Hazard Ratio, $\\exp(\\beta)$", fontweight="bold")
    ax.set_xlim(0.6, 1.4)
    ax.set_ylim(-0.6, len(keep) - 0.15)
    ax.set_title("(b) Relative Reporting Hazard", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.text(0.02, 0.035, "Slower than no injury", transform=ax.transAxes, fontsize=5.6,
            color=fs.MUTED, ha="left")
    ax.text(0.98, 0.035, "Faster", transform=ax.transAxes, fontsize=5.6,
            color=fs.MUTED, ha="right")
    ax.text(0.98, 1.015, "95% CI; reference = no injury", transform=ax.transAxes,
            fontsize=5.45, color=fs.MUTED, ha="right", va="bottom")
    fs.light_grid(ax, axis="x")

    # ---- (c) historical quantile belts
    ax = axes[1, 0]
    for s in (0, 3):
        g = dr[dr.severity == s].sort_values("cohort_year")
        g = g[g["n"] >= 30]
        ax.fill_between(g["cohort_year"], g["p25"], g["p75"],
                        facecolor=fs.SEV_COLORS[s], alpha=0.16, lw=0.0)
        ax.plot(g["cohort_year"], g["p50"], color=fs.SEV_COLORS[s], lw=1.45,
                ls=fs.SEV_LS[s], zorder=3)
        ax.scatter(g["cohort_year"].iloc[::2], g["p50"].iloc[::2],
                   marker=fs.SEV_MARKER[s], s=8, facecolor=fs.SEV_COLORS[s],
                   edgecolor="white", linewidth=0.25, zorder=4)
    ga = dr[dr.severity == -1].sort_values("cohort_year")
    ax.plot(ga["cohort_year"], ga["p50"], color=fs.INK, lw=0.85,
            ls=(0, (1, 1.5)), zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1800)
    ax.set_xlabel("Incident Cohort Year", fontweight="bold")
    ax.set_ylabel("Reporting Delay (Days)", fontweight="bold")
    ax.set_title("(c) Historical Delay Drift", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.legend(handles=[Line2D([0], [0], color=fs.SEV_COLORS[3], lw=1.2,
                              ls=fs.SEV_LS[3], label="fatality"),
                       Line2D([0], [0], color=fs.SEV_COLORS[0], lw=1.2,
                              ls=fs.SEV_LS[0], label="no injury"),
                       Line2D([0], [0], color=fs.INK, lw=0.8, ls=(0, (1, 1.5)),
                              label="all")],
              loc="upper right", fontsize=5.9, ncol=3, columnspacing=0.75,
              handlelength=1.45, handletextpad=0.35)
    ax.text(0.01, 0.035, "Line = median; shaded band = IQR", transform=ax.transAxes,
            fontsize=5.45, color=fs.MUTED, ha="left")
    fs.light_grid(ax)

    # ---- (d) visibility-deficit matrix (replaces a conventional grouped bar chart)
    ax = axes[1, 1]
    hs = [1, 3, 6, 12]
    deficit = np.array([
        [1 - ec[(ec.severity == s) & (ec.d == h)]["F"].iloc[0] for h in hs]
        for s in range(4)
    ])
    ax.imshow(deficit, cmap=fs.HEATMAP_CMAP, vmin=0.0, vmax=0.60,
              aspect="auto", interpolation="nearest")
    for s in range(4):
        for j in range(len(hs)):
            value = deficit[s, j]
            ax.text(j, s, f"{value:.0%}", ha="center", va="center",
                    fontsize=6.85, fontweight="bold",
                    color="#FFFFFF" if value >= 0.30 else fs.INK, zorder=3)
    ax.set_xticks(np.arange(len(hs)))
    ax.set_xticklabels([f"{h}" for h in hs])
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["No injury", "Crash/fire", "Injury", "Fatality"], fontsize=6.25)
    for tick, s in zip(ax.get_yticklabels(), range(4)):
        tick.set_color(fs.SEV_COLORS[s])
        tick.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(hs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Months Since Incident", fontweight="bold")
    ax.set_ylabel("Severity Class", fontweight="bold")
    ax.set_title("(d) Reporting Visibility Deficit", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    # Panel (d) is the EMPIRICAL mature-cohort CDF, not the chain-ladder estimate
    # F_hat used for the correction; the manuscript writes the former F-tilde.
    ax.text(0.98, 1.015, r"Cell = 1 − $\tilde{F}_s(h)$, empirical", transform=ax.transAxes,
            fontsize=5.45, color=fs.MUTED, ha="right", va="bottom")
    fs.save_fig(fig, "fig2")


# ==================================================================== Figure 3
def figure3():
    """Reporting triangle, nowcast gap, and visibility milestones."""
    tri = R("triangle_all.csv")
    uc = R("undercount_curves.csv")
    live = A("live_nowcast.csv")

    fig = plt.figure(figsize=(fs.W2, 2.78))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.03, 1.30, 1.12], wspace=0.50)

    # ---- (a) reporting triangle and censoring frontier
    ax = fig.add_subplot(gs[0])
    last = tri.tail(36).reset_index(drop=True)
    M = last[[f"d{d}" for d in range(D_CAP + 1)]].to_numpy(dtype=float)
    cp = last["cohort_p"].to_numpy()
    unobs = (cp[:, None] + np.arange(D_CAP + 1)[None, :]) > T_CUTOFF
    valid = (~unobs) & (M >= 0)
    log_m = np.zeros_like(M, dtype=float)
    log_m[valid] = np.log10(M[valid] + 1)
    Mm = np.ma.masked_where(~valid, log_m)
    cmap = fs.HEATMAP_CMAP.copy()
    cmap.set_bad("#FFFFFF")
    ax.imshow(Mm, aspect="auto", cmap=cmap, origin="upper",
                   interpolation="nearest")
    bnd_y = np.arange(len(cp))
    bnd_x = T_CUTOFF - cp
    ok = (bnd_x >= 0) & (bnd_x <= D_CAP)
    ax.fill_betweenx(bnd_y[ok], bnd_x[ok] + 0.5, D_CAP + 0.5, step="mid",
                     color="#FAE8E5", alpha=0.92, zorder=1)
    ax.step(bnd_x[ok] + 0.5, bnd_y[ok], where="mid", color=fs.SEV_COLORS[3],
            lw=1.05, zorder=3)
    ax.set_xlabel("Reporting Delay, $d$ (Months)", fontweight="bold")
    ax.set_ylabel("Incident Cohort", fontweight="bold")
    ax.set_yticks([0, len(cp) - 1])
    ax.set_yticklabels([last["cohort"].iloc[0], last["cohort"].iloc[-1]], fontsize=6)
    ax.set_xticks([0, 12, 24, 36])
    ax.text(0.04, 0.95, "Darker = more\nreported complaints", transform=ax.transAxes,
            fontsize=5.45, color="#FFFFFF", ha="left", va="top", fontweight="bold")
    ax.text(0.78, 0.16, "Censored\nat cutoff $T$", transform=ax.transAxes, fontsize=5.55,
            color=fs.SEV_COLORS[3], ha="center", va="center", fontweight="bold")
    ax.set_title("(a) Reporting Triangle", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)

    # ---- (b) observed count, nowcast, and currently hidden component
    ax = fig.add_subplot(gs[1])
    g = live[live.stream == "severe"].sort_values("cohort_p").tail(24)
    x = np.arange(len(g))
    ax.fill_between(x, g["lo90"], g["hi90"], color=fs.NOWCAST, alpha=0.16, lw=0)
    ax.fill_between(x, g["lo50"], g["hi50"], color=fs.NOWCAST, alpha=0.30, lw=0)
    ax.fill_between(x, g["C_obs"], g["N_hat"], where=g["N_hat"] >= g["C_obs"],
                    color=fs.SEV_COLORS[2], alpha=0.13, lw=0, zorder=2)
    ax.plot(x, g["N_hat"], color=fs.NOWCAST, lw=1.45, label="nowcast", zorder=4)
    ax.plot(x, g["C_obs"], color="#9A9A9A", lw=1.1, ls=(0, (3, 1.6)),
            label="observed count", zorder=4)
    tk = np.arange(0, len(g), 6)
    ax.set_xticks(tk)
    ax.set_xticklabels([g["cohort"].iloc[i] for i in tk], fontsize=5.8)
    latest = g.iloc[-1]
    gap = latest["N_hat"] - latest["C_obs"]
    x_last = len(g) - 1
    ax.annotate("", xy=(x_last + 0.38, latest["N_hat"]),
                xytext=(x_last + 0.38, latest["C_obs"]),
                arrowprops=dict(arrowstyle="<->", lw=0.75, color=fs.SEV_COLORS[2]))
    ax.text(x_last - 0.15, (latest["N_hat"] + latest["C_obs"]) / 2,
            f"{gap:.0f}\ncurrently unseen", ha="right", va="center", fontsize=5.45,
            color=fs.SEV_COLORS[2], fontweight="bold")
    ax.set_xlim(-0.45, len(g) - 0.05)
    ax.set_ylabel("Severe Complaints (Y$\\geq$1)", fontweight="bold")
    ax.set_xlabel("Incident Cohort Month", fontweight="bold")
    ax.legend(fontsize=5.75, loc="upper left", borderaxespad=0.35,
              handlelength=1.55, handletextpad=0.35)
    ax.set_title("(b) Severe-Complaint Nowcast", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    fs.light_grid(ax)

    # ---- (c) reporting-visibility milestone timeline
    ax = fig.add_subplot(gs[2])
    u = uc[uc["W"] == "60"]
    thresholds = (0.50, 0.75, 0.90, 0.95)
    markers = ("o", "s", "D", "^")
    rows = {0: 3, 1: 2, 2: 1, 3: 0}
    for s in range(4):
        g = u[u.stream == f"Y{s}"].sort_values("d")
        times = [int(g.loc[g["F"] >= q, "d"].iloc[0]) for q in thresholds]
        y = rows[s]
        ax.hlines(y, 0, times[-1], color=fs.SEV_COLORS[s], alpha=0.38,
                  lw=3.2, zorder=1)
        for t, marker in zip(times, markers):
            ax.scatter(t, y, marker=marker, s=33, facecolor=fs.SEV_COLORS[s],
                       edgecolor="white", linewidth=0.55, zorder=3)
    ax.set_xlim(-0.5, 24.8); ax.set_ylim(-0.55, 3.55)
    ax.set_xticks([0, 3, 6, 12, 18, 24])
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(["No injury", "Crash/fire", "Injury", "Fatality"], fontsize=6.15)
    for tick, s in zip(ax.get_yticklabels(), (0, 1, 2, 3)):
        tick.set_color(fs.SEV_COLORS[s])
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Months Required to Reach Target Visibility", fontweight="bold")
    ax.set_ylabel("Severity Class", fontweight="bold")
    ax.set_title("(c) Visibility Milestones", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.text(0.01, 0.04,
            "Left-to-right markers: 50%, 75%, 90%, and 95% visibility",
            transform=ax.transAxes, fontsize=5.3, color=fs.MUTED, ha="left", va="bottom")
    fs.light_grid(ax, axis="x")
    fs.save_fig(fig, "fig3")


# ==================================================================== Figure 4
def figure4():
    """Backtest calibration, interval coverage, and case-study detection lead."""
    obs = A("backtest_obs.csv")
    met = A("backtest_metrics.csv")
    tr = R("case_z_traces.csv")
    with open(os.path.join(AUDIT_DIR, "case_studies.json"), encoding="utf-8") as fh:
        cs = json.load(fh)

    ev = obs[obs["Tstar"] >= str_to_p("2021-01")]
    mev = met[met["origin_set"].str.startswith("evaluation")]

    fig = plt.figure(figsize=(fs.W2, 5.35))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.15],
        height_ratios=[0.92, 1.16],
        hspace=0.34,
        wspace=0.52,
    )

    # ---- (a) calibration residual field
    ax = fig.add_subplot(gs[0, 0])
    ax.axhspan(0.80, 1.20, color=fs.NOWCAST, alpha=0.075, zorder=0)
    ax.axhline(1.0, color=fs.INK, lw=0.65, ls=(0, (3, 2)), zorder=1)
    for s in range(4):
        g = ev[(ev.stream == f"Y{s}") & (ev.realized > 0)].copy()
        ratio = g["N_hat"] / g["realized"]
        good = np.isfinite(ratio) & (ratio > 0)
        ax.plot(g.loc[good, "realized"], ratio.loc[good], fs.SEV_MARKER[s], ms=2.0,
                alpha=0.43, color=fs.SEV_COLORS[s], mec="none", ls="none",
                label=SHORT[s], zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.6, 6e4); ax.set_ylim(0.20, 5.0)
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Realized Cohort Count", fontweight="bold")
    ax.set_ylabel("Nowcast / Realized", fontweight="bold")
    ax.legend(fontsize=5.65, ncol=2, loc="upper left", markerscale=2.4,
              handletextpad=0.2, columnspacing=0.65, borderaxespad=0.35)
    ax.set_title("(a) Backtest Calibration Residuals", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.text(0.98, 0.53, "Exact\ncalibration", transform=ax.transAxes, fontsize=5.45,
            color=fs.MUTED, ha="right", va="bottom")
    fs.light_grid(ax)

    # ---- (b) interval-coverage calibration matrix
    ax = fig.add_subplot(gs[0, 1])
    nc = mev[mev.method == "nowcast"]
    coverage_rows, row_labels, row_severity, targets = [], [], [], []
    for target, key in ((0.90, "cov90"), (0.50, "cov50")):
        for s in range(4):
            g = nc[nc.stream == f"Y{s}"].set_index("h")
            coverage_rows.append([float(g.loc[h, key]) for h in HORIZONS])
            row_labels.append(f"{int(target * 100)}% | {SHORT[s]}")
            row_severity.append(s)
            targets.append(target)
    coverage = np.asarray(coverage_rows)
    deviation = coverage - np.asarray(targets)[:, None]
    coverage_cmap = fs.mpl.colors.LinearSegmentedColormap.from_list(
        "coverage_deviation", ["#C84A38", "#F8F4F2", fs.NOWCAST]
    )
    coverage_norm = fs.mpl.colors.TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.10)
    ax.imshow(deviation, cmap=coverage_cmap, norm=coverage_norm,
              aspect="auto", interpolation="nearest")
    for i in range(coverage.shape[0]):
        for j in range(coverage.shape[1]):
            value, dev = coverage[i, j], deviation[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=5.7,
                    fontweight="bold", color="#FFFFFF" if abs(dev) >= 0.075 else fs.INK)
    ax.axhline(3.5, color="#FFFFFF", lw=2.0)
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.25)
    for tick, s in zip(ax.get_yticklabels(), row_severity):
        tick.set_color(fs.SEV_COLORS[s])
        tick.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(HORIZONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel("Forecast Horizon, $h$ (Months)", fontweight="bold")
    ax.set_ylabel("Nominal Interval / Severity", fontweight="bold")
    ax.set_title("(b) Interval Coverage", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)

    # ---- (c) pseudo-real-time detection lead in the primary case study
    ax = fig.add_subplot(gs[1, :])
    case = cs["primary_cases"][0]
    g = tr[tr["case"] == case["name"]].sort_values("Tstar").reset_index(drop=True)
    x = np.arange(len(g))
    z_min = min(g["z_naive"].min(), g["z_naive_matched"].min(), g["z_nowcast"].min())
    z_max = max(g["z_naive"].max(), g["z_naive_matched"].max(), g["z_nowcast"].max())
    ax.axhspan(3.0, z_max + 1.4, color=fs.SEV_COLORS[3], alpha=0.045, zorder=0)
    ax.plot(x, g["z_naive"], color="#9A9A9A", lw=1.05, ls=(0, (3, 1.6)), label="naive")
    ax.plot(x, g["z_naive_matched"], color=fs.SEV_COLORS[2], lw=1.05, ls=(0, (1, 1.4)),
            label="naive, maturity-matched")
    ax.plot(x, g["z_nowcast"], color=fs.NOWCAST, lw=1.55, label="nowcast", zorder=4)
    ax.axhline(3.0, color=fs.SEV_COLORS[3], lw=0.75, ls=(0, (4, 2)))
    ax.text(len(g) - 0.5, 3.25, "Alarm threshold, z = 3", fontsize=5.55,
            color=fs.SEV_COLORS[3],
            ha="right")
    pos = {m: i for i, m in enumerate(g["Tstar"])}
    for arm, col in (("nowcast", fs.NOWCAST), ("naive_matched", fs.SEV_COLORS[2]),
                     ("naive", "#7A7A7A")):
        tp = case.get(f"tau_{arm}_p")
        if tp in pos:
            ax.scatter(pos[tp], g[f"z_{arm}"].iloc[pos[tp]], marker="v", s=32,
                       color=col, edgecolor="#FFFFFF", linewidth=0.45, zorder=5)
    tn = case.get("tau_nowcast_p")
    tm = case.get("tau_naive_matched_p")
    tw = case.get("tau_naive_p")
    ax.set_ylim(z_min - 0.60, z_max + 2.35)
    # Each alarm month carries its own label, tied to its own marker by a leader line,
    # so the reader never has to guess which trace a floating annotation refers to.
    d_m = case["delta_vs_naive_matched_months"]
    d_n = case["delta_vs_naive_months"]
    labels = [
        ("nowcast", tn, fs.NOWCAST, f"nowcast alarms\n{case['tau_nowcast']}",
         (0.6, z_max + 1.55)),
        ("naive_matched", tm, fs.SEV_COLORS[2],
         f"maturity-matched\n{case['tau_naive_matched']}  (+{d_m} mo)",
         (10.2, z_max + 1.55)),
        ("naive", tw, "#6E6E6E",
         f"raw naive\n{case['tau_naive']}  (+{d_n} mo)",
         (24.5, z_max + 1.55)),
    ]
    for arm, tp, col, text, (tx, ty) in labels:
        if tp not in pos:
            continue
        ax.annotate(
            text, xy=(pos[tp], g[f"z_{arm}"].iloc[pos[tp]]), xytext=(tx, ty),
            ha="left", va="top", fontsize=5.6, fontweight="bold", color=col,
            linespacing=1.3, zorder=6,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=col, alpha=0.85,
                            shrinkA=2, shrinkB=3,
                            connectionstyle="angle,angleA=0,angleB=90,rad=2"))
    tk = np.arange(0, len(g), 6)
    ax.set_xticks(tk)
    ax.set_xticklabels([g["month"].iloc[i] for i in tk], fontsize=5.9,
                       rotation=45, ha="right")
    ax.set_ylabel("Detection Score, $z_j(t)$", fontweight="bold")
    ax.set_xlabel("Monitoring Month (Pseudo-Real-Time)", fontweight="bold")
    ax.legend(fontsize=5.9, loc="upper right", bbox_to_anchor=(0.995, 0.83),
              borderaxespad=0.35, handlelength=1.7, handletextpad=0.4)
    ax.set_title("(c) Case-Study Detection Lead", loc="left", fontsize=7.4,
                 fontweight="bold", pad=7)
    ax.text(0.015, 0.035, case["name"], transform=ax.transAxes, fontsize=6.05,
            color=fs.MUTED, ha="left", va="bottom")
    fs.light_grid(ax)

    # The stream-wide, specificity-matched detection-gain distribution is reported in
    # the Results text rather than as a fourth panel; it is a single spike at zero and
    # a histogram of it added nothing a sentence does not carry.

    fs.save_fig(fig, "fig4")


def main():
    for fn in (figure1, figure2, figure3, figure4):
        fn()
        print(f"  {fn.__name__} written", flush=True)
    print("  figures + 600-dpi PNGs + grayscale proofs complete")


if __name__ == "__main__":
    main()
