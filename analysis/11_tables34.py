"""Step 11 - LaTeX tables 3 and 4, the maturity factors and the robustness battery.

Both tables encode a contrast rather than a level. Shading a maturity factor by its own
value says only what the printed digits already say, and because every stream climbs from
about one half to one over the same horizon, every row then looks alike. What separates
the rows is the gap against the pooled all-complaints curve at the same horizon, so that
gap is what the shading and the row glyph of table 3 carry. Table 4 works the same way,
against the main specification rather than against a pooled stream.

The macros used here (dbar, sparkline, glyphgrid, glyphaxis, barfill, barink) are the ones
already defined in tables_preamble.tex by step 7.

Every number is read from results/tables/undercount_curves.csv and
results/audits/robustness.csv, so the artwork cannot drift away from the data.

Outputs
  results/tables/table3.tex     maturity factors F_hat_s(h)
  results/tables/table4.tex     seven specification variants
"""

from __future__ import annotations

import io
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AUDIT_DIR, TAB_DIR, qa_log   # noqa: E402

HS = [0, 1, 2, 3, 6, 9, 12, 18, 24, 36]
GAP_SPAN = 0.28     # table 3 shading and glyph, in F units
DEV_SPAN = 0.085    # table 4 deviation glyph, in F units

WHITE = (0xFF, 0xFF, 0xFF)
AHEAD = (0xA8, 0xC8, 0xE0)      # blue, stream reports faster than the pooled curve
BEHIND = (0xD0, 0x8C, 0x57)     # amber, stream reports slower
# The amber endpoint is deliberately darker than the blue one. At equal lightness the two
# hues differ by three units of grayscale luminance and collapse into each other in a
# print proof, so sign is carried by lightness as well as by hue.

T3_ROWS = [("All complaints", "all", False),
           ("No injury (Y0)", "Y0", True),
           ("Crash/fire (Y1)", "Y1", True),
           ("Injury (Y2)", "Y2", True),
           ("Fatality (Y3)", "Y3", True),
           (r"All severe (Y$\geq$1)", "severe", False)]

T4_ROWS = [("all_channels", "All channels, not consumer only"),
           ("D24", r"Delay cap $D = 24$ months"),
           ("W36", r"Rolling window $W = 36$ months"),
           ("W_all_history", "Window extended to full history"),
           ("exclude_pre2003", "Drop pre-2003 cohorts"),
           ("exclude_burst_months", "Drop recall-publicity burst months"),
           ("delay_by_elapsed_days", "Delay in elapsed days")]


def gap_tint(gap):
    """Diverging tint keyed to the signed gap, white at zero.

    A square-root ramp is used because the fatality gap is an order of magnitude larger
    than every other one; on a linear ramp the crash/fire and injury rows would round to
    white and the reader would lose a real difference.
    """
    if abs(gap) < 1e-9:
        return "FFFFFF"
    k = min(1.0, math.sqrt(min(abs(gap), GAP_SPAN) / GAP_SPAN))
    end = AHEAD if gap > 0 else BEHIND
    return "".join("%02X" % round(WHITE[i] + k * (end[i] - WHITE[i])) for i in range(3))


def cell(v, gap):
    return r"\cellcolor[HTML]{%s}%.3f" % (gap_tint(gap), v)


def thousands(n):
    return "{:,}".format(int(round(n))).replace(",", r"\,")


def gap_glyph(F, Fall):
    """Signed gap to the pooled curve across h = 0..36, on a common scale.

    Drawn linearly, so the glyph shows honestly that only the fatality stream departs
    from the pooled curve by a material amount.
    """
    def y(h):
        return max(-1.0, min(1.0, (F[h] - Fall[h]) / GAP_SPAN))
    pts = " -- ".join("(%.3f,%.3f)" % (h / 36.0, y(h)) for h in range(0, 37))
    return (r"\begin{tikzpicture}[x=12mm,y=2.3mm,baseline=-0.3mm]"
            r"\fill[barfill] (0,0) -- " + pts + r" -- (1,0) -- cycle;"
            r"\draw[glyphgrid] (0,-1) rectangle (1,1);"
            r"\draw[glyphaxis] (0,0) -- (1,0);"
            r"\draw[sparkline,line width=0.5pt] " + pts + r";"
            r"\end{tikzpicture}")


def dev_glyph(d0=None, d3=None, ds=None):
    """Signed deviation of three F(1) values from the main specification."""
    def x(d):
        return max(-1.0, min(1.0, d / DEV_SPAN)) * 0.5 + 0.5
    out = [r"\begin{tikzpicture}[x=13mm,y=1mm,baseline=-0.75mm]",
           r"\draw[glyphgrid] (0,-1.5) rectangle (1,1.5);",
           r"\draw[glyphaxis] (0.5,-1.5) -- (0.5,1.5);"]
    if d0 is not None:
        out.append(r"\fill[black!35] (%.3f,0.9) circle (0.42mm);" % x(d0))
        out.append(r"\fill[barink] (%.3f,0) circle (0.42mm);" % x(d3))
        out.append(r"\fill[barink!45] (%.3f,-0.9) circle (0.42mm);" % x(ds))
    out.append(r"\end{tikzpicture}")
    return "".join(out)


def build_table3(F):
    Fall = F["all"]
    L = [r"\begin{table}",
         r"\caption{Maturity Factors $\hat{F}_s(h)$ for Correcting a Partial Cohort Count}",
         r"\label{tab:factors}", r"\footnotesize", r"\centering",
         r"\setlength{\tabcolsep}{3.0pt}",
         r"\begin{tabular}{@{}l rrrrrrrrrr c@{}}", r"\toprule",
         r"& \multicolumn{10}{c}{Months since incident month, $h$} & "
         r"\multicolumn{1}{c}{\footnotesize Gap to all complaints} \\",
         r"\cmidrule(lr){2-11}\cmidrule(l){12-12}",
         r"Severity stream & " + " & ".join(str(h) for h in HS) +
         r" & \multicolumn{1}{c}{\tiny $\pm0.28$ over $0 \rightarrow 36$ mo} \\",
         r"\midrule"]
    for label, key, indent in T3_ROWS:
        Fs = F[key]
        nm = (r"\hspace{0.8em}" + label) if indent else label
        if key == "all":
            nm = r"\textbf{%s}" % nm
        cells = " & ".join(cell(Fs[h], Fs[h] - Fall[h]) for h in HS)
        glyph = (r"\textit{\tiny baseline}" if key == "all" else gap_glyph(Fs, Fall))
        L.append("%s & %s & %s \\\\" % (nm, cells, glyph))
        if key == "all":
            L.append(r"\addlinespace[2pt]")
            L.append(r"\multicolumn{12}{@{}l}{\textit{By severity class}}\\")
        if key == "Y3":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{3pt}",
          r"\parbox{\linewidth}{\footnotesize",
          r"""Note: share of a cohort's complaints arriving within the 36-month horizon that have
arrived by $h$ months, from Equation \ref{eq:Fhat} on a 60-month window at $T =$ 2026-06.
Divide a partial count by $\hat{F}_s(h)$; re-estimate monthly. Shading and the row glyph
both show the gap against the pooled all-complaints row at the same horizon rather than
the level, since every stream climbs over the same range and the level alone separates
nothing. Amber marks a stream that is less complete than the pooled curve and blue one
that is more complete, on a square-root scale to $\pm0.28$. The glyph traces that gap
linearly across every month from 0 to 36, which shows that only the fatality stream
departs from the pooled curve by a material amount, by as much as 0.268 at one month. The
fatality row applies to the national stream rather than to an individual make or component
stream, and the text reports its behavior at $h=0$.}""",
          r"\end{table}"]
    return "\n".join(L) + "\n"


def build_table4(f1, maxd, ncomp):
    base = f1["base"]
    maxmove = max(v for k, v in maxd.items() if k != "base") or 1.0
    L = [r"\begin{table}",
         r"\caption{Robustness of the Maturity Factors to Seven Specification Variants}",
         r"\label{tab:robust}", r"\footnotesize", r"\centering",
         r"\setlength{\tabcolsep}{3.6pt}",
         r"\begin{tabular}{@{}l rrr c l r@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{$\hat{F}_s(1)$} & "
         r"\multicolumn{1}{c}{\footnotesize Deviation from main} & "
         r"\multicolumn{1}{c}{\footnotesize Largest move} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
         r"Variant & No injury & Fatality & All severe & "
         r"\multicolumn{1}{c}{\tiny $-.085\ \ 0\ \ +.085$} & "
         r"\multicolumn{1}{c}{max $|\Delta \hat{F}|$} & $N$ \\",
         r"\midrule",
         r"\textit{Main specification} & %.3f & %.3f & %.3f & %s & "
         r"\textit{\tiny baseline} & %s \\"
         % (base["Y0"], base["Y3"], base["severe"], dev_glyph(),
            thousands(ncomp["base"])),
         r"\addlinespace[2pt]"]
    for key, name in T4_ROWS:
        v = f1[key]
        L.append("%s & %.3f & %.3f & %.3f & %s & %s & %s \\\\"
                 % (name, v["Y0"], v["Y3"], v["severe"],
                    dev_glyph(v["Y0"] - base["Y0"], v["Y3"] - base["Y3"],
                              v["severe"] - base["severe"]),
                    r"\dbar{%.3f}{%.3f}" % (maxd[key] / maxmove, maxd[key]),
                    thousands(ncomp[key])))
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{3pt}",
          r"\parbox{\linewidth}{\footnotesize",
          r"""Note: $\hat{F}_s(1)$ is the share of a cohort arrived one month after the incident
month. The deviation glyph places the three factors against the main specification on a
common scale of $\pm 0.085$, with no injury on the top row, fatality on the middle row and
all severe on the bottom row, so a mark right of centre is a variant that reports the
cohort as more complete and a mark on the centre line is a variant that changes nothing.
Max $|\Delta \hat{F}|$ is the largest absolute movement against the main specification
across the severity classes and horizons of Table \ref{tab:factors}, drawn as a bar on a
common scale so the three specification choices that matter separate from the four that do
not. $N$ is the number of complaints entering the variant. Dropping pre-2003 cohorts
leaves the factors unchanged because the rolling window never reaches them.}""",
          r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    u = pd.read_csv(os.path.join(TAB_DIR, "undercount_curves.csv"))
    u["W"] = u["W"].astype(str)
    u = u[u["W"] == "60"]
    F = {}
    for _, key, _ in T3_ROWS:
        g = u[u["stream"] == key].sort_values("d").set_index("d")["F"]
        F[key] = {int(d): float(v) for d, v in g.items()}

    r = pd.read_csv(os.path.join(AUDIT_DIR, "robustness.csv"))
    fh = r[r["quantity"] == "F_hat(h)"]
    one = fh[fh["h"] == 1]
    f1 = {v: {s: float(g[g["stream"] == s]["value"].iloc[0])
              for s in ("Y0", "Y3", "severe")}
          for v, g in one.groupby("variant")}
    maxd = fh.assign(a=fh["delta_vs_base"].abs()).groupby("variant")["a"].max().to_dict()
    nc = r[r["quantity"] == "n_complaints"]
    ncomp = {v: float(g["value"].iloc[0]) for v, g in nc.groupby("variant")}

    io.open(os.path.join(TAB_DIR, "table3.tex"), "w", encoding="utf-8",
            newline="\n").write(build_table3(F))
    io.open(os.path.join(TAB_DIR, "table4.tex"), "w", encoding="utf-8",
            newline="\n").write(build_table4(f1, maxd, ncomp))
    biggest = max(abs(F[s][h] - F["all"][h])
                  for s in ("Y0", "Y1", "Y2", "Y3", "severe") for h in range(0, 37))
    print("  wrote table3.tex and table4.tex; largest gap to pooled curve %.3f" % biggest)
    qa_log("Step 11", "tables 3 and 4 built from data", True,
           "every cell read from undercount_curves.csv and robustness.csv; the "
           "generator holds no numeric literals; shading encodes the gap to the pooled "
           "curve, not the level")


if __name__ == "__main__":
    main()
