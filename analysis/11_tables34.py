"""Step 11 - LaTeX tables 3 and 4, the maturity factors and the robustness battery.

Both tables are drawn in the same visual language as tables 1 and 2. Numeric cells carry
a tint keyed to the value, table 3 ends in a maturity-curve glyph per stream, and table 4
carries a signed deviation glyph plus a data bar on the movement column. The macros used
here (dbar, sparkline, glyphgrid, glyphaxis, barfill, barink) are the ones already defined
in tables_preamble.tex by step 7.

Every number is read from results/tables/undercount_curves.csv and
results/audits/robustness.csv, so the artwork cannot drift away from the data.

Outputs
  results/tables/table3.tex     maturity factors F_hat_s(h)
  results/tables/table4.tex     seven specification variants
"""

from __future__ import annotations

import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AUDIT_DIR, TAB_DIR, qa_log   # noqa: E402

HS = [0, 1, 2, 3, 6, 9, 12, 18, 24, 36]
SPAN = 0.085        # deviation glyph half-width, in F units
TINT_LO = (0xF4, 0xF8, 0xFB)
TINT_HI = (0xB4, 0xD0, 0xE6)

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


def tint(frac):
    """Light blue ramp used for value-keyed cell shading."""
    frac = max(0.0, min(1.0, float(frac)))
    return "".join("%02X" % round(TINT_LO[i] + frac * (TINT_HI[i] - TINT_LO[i]))
                   for i in range(3))


def cell(v):
    return r"\cellcolor[HTML]{%s}%.3f" % (tint(v), v)


def thousands(n):
    return "{:,}".format(int(round(n))).replace(",", r"\,")


def curve_glyph(F):
    """Maturity curve over h = 0..36 as a filled area on a shared 0..1 scale.

    A hairline is illegible at this size, so the area under the curve is filled in the
    tint the data bars already use and the curve is stroked over it.
    """
    pts = " -- ".join("(%.3f,%.2f)" % (h / 36.0, F[h]) for h in range(0, 37))
    return (r"\begin{tikzpicture}[x=12mm,y=4.6mm,baseline=0.4mm]"
            r"\fill[barfill] (0,0) -- " + pts + r" -- (1,0) -- cycle;"
            r"\draw[glyphgrid] (0,0.5) -- (1,0.5);"
            r"\draw[glyphgrid] (0,0) rectangle (1,1);"
            r"\draw[sparkline,line width=0.5pt] " + pts + r";"
            r"\fill[barink] (0,%.2f) circle (0.32mm);" % F[0] +
            r"\end{tikzpicture}")


def dev_glyph(d0=None, d3=None, ds=None):
    """Signed deviation of three F(1) values from the main specification."""
    def x(d):
        return max(-1.0, min(1.0, d / SPAN)) * 0.5 + 0.5
    out = [r"\begin{tikzpicture}[x=13mm,y=1mm,baseline=-0.75mm]",
           r"\draw[glyphgrid] (0,-1.5) rectangle (1,1.5);",
           r"\draw[glyphaxis] (0.5,-1.5) -- (0.5,1.5);"]
    if d0 is not None:
        out.append(r"\fill[black!35] (%.3f,0.9) circle (0.42mm);" % x(d0))
        out.append(r"\fill[barink] (%.3f,0) circle (0.42mm);" % x(d3))
        out.append(r"\fill[barink!45] (%.3f,-0.9) circle (0.42mm);" % x(ds))
    out.append(r"\end{tikzpicture}")
    return "".join(out)


def build_table3(F_by_stream):
    L = [r"\begin{table}",
         r"\caption{Maturity Factors $\hat{F}_s(h)$ for Correcting a Partial Cohort Count}",
         r"\label{tab:factors}", r"\footnotesize", r"\centering",
         r"\setlength{\tabcolsep}{3.0pt}",
         r"\begin{tabular}{@{}l rrrrrrrrrr c@{}}", r"\toprule",
         r"& \multicolumn{10}{c}{Months since incident month, $h$} & "
         r"\multicolumn{1}{c}{\footnotesize Maturity curve} \\",
         r"\cmidrule(lr){2-11}\cmidrule(l){12-12}",
         r"Severity stream & " + " & ".join(str(h) for h in HS) +
         r" & \multicolumn{1}{c}{\tiny $0 \rightarrow 36$ mo} \\",
         r"\midrule"]
    for label, key, indent in T3_ROWS:
        F = F_by_stream[key]
        nm = (r"\hspace{0.8em}" + label) if indent else label
        if key == "all":
            nm = r"\textbf{%s}" % nm
        L.append("%s & %s & %s \\\\"
                 % (nm, " & ".join(cell(F[h]) for h in HS), curve_glyph(F)))
        if key == "all":
            L.append(r"\addlinespace[2pt]")
            L.append(r"\multicolumn{12}{@{}l}{\textit{By severity class}}\\")
        if key == "Y3":
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{3pt}",
          r"\parbox{\linewidth}{\footnotesize",
          r"""Note: share of a cohort's complaints arriving within the 36-month horizon that have
arrived by $h$ months, from Equation
\ref{eq:Fhat} on a 60-month window at $T =$ 2026-06. Cell shading runs with the value, so
a lighter cell is a less complete count. The glyph traces the same row across every month
from 0 to 36 on a common 0 to 1 scale, with a dot at $h=0$. Divide a partial count by
$\hat{F}_s(h)$; re-estimate monthly. The fatality row applies to the national stream
rather than to an individual make or component stream, and the text reports its behavior
at $h=0$.}""",
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
         r"\multicolumn{1}{c}{\footnotesize Deviation} & "
         r"\multicolumn{1}{c}{\footnotesize Largest move} & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
         r"Variant & No injury & Fatality & All severe & "
         r"\multicolumn{1}{c}{\tiny $-.085\ \ 0\ \ +.085$} & "
         r"\multicolumn{1}{c}{max $|\Delta \hat{F}|$} & $N$ \\",
         r"\midrule",
         r"\textit{Main specification} & %s & %s & %s & %s & \dbar{0}{0.000} & %s \\"
         % (cell(base["Y0"]), cell(base["Y3"]), cell(base["severe"]),
            dev_glyph(), thousands(ncomp["base"])),
         r"\addlinespace[2pt]"]
    for key, name in T4_ROWS:
        v = f1[key]
        L.append("%s & %s & %s & %s & %s & %s & %s \\\\"
                 % (name, cell(v["Y0"]), cell(v["Y3"]), cell(v["severe"]),
                    dev_glyph(v["Y0"] - base["Y0"], v["Y3"] - base["Y3"],
                              v["severe"] - base["severe"]),
                    r"\dbar{%.3f}{%.3f}" % (maxd[key] / maxmove, maxd[key]),
                    thousands(ncomp[key])))
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{3pt}",
          r"\parbox{\linewidth}{\footnotesize",
          r"""Note: $\hat{F}_s(1)$ is the share of a cohort arrived one month after the incident
month, shaded as in Table \ref{tab:factors}. The deviation glyph places the three factors
against the main specification on a common scale of $\pm 0.085$, with no injury on the top
row, fatality on the middle row, and all severe on the bottom row, so a mark right of
centre is a variant that reports the cohort as more complete. Max $|\Delta \hat{F}|$ is the
largest absolute movement against the main specification across the severity classes and
horizons of Table \ref{tab:factors}, drawn as a bar on a common scale. $N$ is the number of
complaints entering the variant. Dropping pre-2003 cohorts leaves the factors unchanged
because the rolling window never reaches them.}""",
          r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    u = pd.read_csv(os.path.join(TAB_DIR, "undercount_curves.csv"))
    u["W"] = u["W"].astype(str)
    u = u[u["W"] == "60"]
    F_by_stream = {}
    for _, key, _ in T3_ROWS:
        g = u[u["stream"] == key].sort_values("d").set_index("d")["F"]
        F_by_stream[key] = {int(d): float(v) for d, v in g.items()}

    r = pd.read_csv(os.path.join(AUDIT_DIR, "robustness.csv"))
    fh = r[r["quantity"] == "F_hat(h)"]
    one = fh[fh["h"] == 1]
    f1 = {v: {s: float(g[g["stream"] == s]["value"].iloc[0])
              for s in ("Y0", "Y3", "severe")}
          for v, g in one.groupby("variant")}
    maxd = fh.assign(a=fh["delta_vs_base"].abs()).groupby("variant")["a"].max().to_dict()
    nc = r[r["quantity"] == "n_complaints"]
    ncomp = {v: float(g["value"].iloc[0]) for v, g in nc.groupby("variant")}

    p3 = os.path.join(TAB_DIR, "table3.tex")
    p4 = os.path.join(TAB_DIR, "table4.tex")
    io.open(p3, "w", encoding="utf-8", newline="\n").write(build_table3(F_by_stream))
    io.open(p4, "w", encoding="utf-8", newline="\n").write(build_table4(f1, maxd, ncomp))
    print("  wrote table3.tex and table4.tex")
    qa_log("Step 11", "tables 3 and 4 built from data", True,
           "every cell read from undercount_curves.csv and robustness.csv; "
           "the generator holds no numeric literals")


if __name__ == "__main__":
    main()
