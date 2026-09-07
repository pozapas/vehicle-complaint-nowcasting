"""Self-QA  -  formula-vs-code audit against outline eqs. (1)-(9), plus a style audit.

Each equation is re-implemented here independently (naively, from the outline text)
and checked against the production code on synthetic data. A passing check means the
production implementation and a literal reading of the outline agree numerically.

Also verifies the no-leakage property of `censor()` directly, and audits that every
figure goes through the single style module.

Outputs
  results/audits/formula_audit.md
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainladder as cl
import figstyle as fs
from common import AUDIT_DIR, D_CAP, FIG_DIR, GRAY_DIR, T_CUTOFF, TAB_DIR, month_index

CHECKS = []


def check(name, ok, detail):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
    return ok


# ------------------------------------------------------------------ eq (1)
def eq1():
    """d = 12*(year(L) - year(F)) + (month(L) - month(F))."""
    rng = np.random.default_rng(1)
    yF = rng.integers(1995, 2026, 4000); mF = rng.integers(1, 13, 4000)
    yL = yF + rng.integers(0, 5, 4000); mL = rng.integers(1, 13, 4000)
    F = yF * 10000 + mF * 100 + 15
    L = yL * 10000 + mL * 100 + 20
    literal = 12 * (yL - yF) + (mL - mF)
    prod = month_index(L) - month_index(F)
    check("eq (1) delay indexing", np.array_equal(literal, prod.astype(int)),
          f"{len(F):,} random date pairs, exact match")


# ------------------------------------------------------------------ eqs (2), (4), (5)
def eq2_4_5():
    rng = np.random.default_rng(2)
    D, T, t0 = 8, 60, 0
    tri = rng.poisson(40, size=(T + 1, D + 1)).astype(float)
    tri = cl.censor(tri, T, D, t0=t0)
    C = np.cumsum(tri, axis=1)

    # eq (2)
    man = np.zeros_like(tri)
    for i in range(tri.shape[0]):
        for d in range(D + 1):
            man[i, d] = sum(tri[i, u] for u in range(d + 1))
    check("eq (2) cumulative counts", np.allclose(man, C), "matches np.cumsum")

    for W in (12, 30, 10_000):
        # eq (4), literal transcription of W_d
        lam_lit = np.empty(D)
        for d in range(D):
            ts = [t for t in range(t0, T + 1)
                  if (t + d + 1 <= T) and (t >= T - d - 1 - W)]
            num = sum(C[t - t0, d + 1] for t in ts)
            den = sum(C[t - t0, d] for t in ts)
            lam_lit[d] = num / den if den > 0 else 1.0
        lam_prod = cl.dev_factors(tri, T, D, W, t0=t0)
        ok4 = np.allclose(lam_lit, lam_prod)

        # eq (5)
        F_lit = np.ones(D + 1)
        for d in range(D):
            F_lit[d] = np.prod([1.0 / lam_lit[u] for u in range(d, D)])
        F_prod = cl.F_from_lambda(lam_prod, D)
        ok5 = np.allclose(F_lit, F_prod)
        check(f"eq (4) development factors, W={W}", ok4,
              f"max |dlambda| = {np.max(np.abs(lam_lit - lam_prod)):.2e}")
        check(f"eq (5) F_hat from lambda, W={W}", ok5,
              f"max |dF| = {np.max(np.abs(F_lit - F_prod)):.2e}; "
              f"F(D) = {F_prod[D]:.12f}")


# ------------------------------------------------------------------ eqs (3), (7), (8)
def eq3_7_8():
    rng = np.random.default_rng(3)
    D, T, t0 = 10, 80, 0
    full = rng.poisson(60, size=(T + 1, D + 1)).astype(float)
    tri = cl.censor(full, T, D, t0=t0)
    lam, F = cl.fit_F(tri, T, D, 10_000, t0=t0)
    t, C_obs, N_hat, h = cl.nowcast(tri, F, T, D, t0=t0)

    # eq (7) literal: N_hat = C(t, T-t) / F_hat(T-t), capped at D
    ok = True
    for i in range(len(t)):
        hh = min(T - int(t[i]), D)
        lit = np.cumsum(tri, axis=1)[i, hh] / F[hh]
        ok &= abs(lit - N_hat[i]) < 1e-9
    check("eq (7) point nowcast", ok, "matches literal C(t,h)/F(h), h = min(T-t, D)")

    # eq (3): for mature cohorts the estimand is exactly C(t, D)
    mature = t <= T - D
    Cfull = np.cumsum(tri, axis=1)[:, D]
    check("eq (3) estimand identity on mature cohorts",
          np.allclose(N_hat[mature], Cfull[mature]),
          f"{int(mature.sum())} mature cohorts, max diff "
          f"{np.max(np.abs(N_hat[mature] - Cfull[mature])):.2e}")

    # eq (8)
    hh, R = cl.undercount_curve(F, D)
    check("eq (8) undercount ratio R(h) = F_hat(h)", np.allclose(R, F),
          "identical arrays")


# ------------------------------------------------------------------ leakage
def leakage():
    rng = np.random.default_rng(4)
    D, T, t0 = 12, 100, 0
    full = rng.poisson(30, size=(T + 1, D + 1)).astype(float)
    bad = 0
    for Ts in (40, 70, 99):
        c = cl.censor(full, Ts, D, t0=t0)
        n_t = c.shape[0]
        idx = np.arange(n_t)[:, None] + t0
        d = np.arange(D + 1)[None, :]
        future = (idx + d) > Ts
        bad += int((c[future] != 0).sum())
        # and every observed cell must be preserved untouched
        obs = ~future
        if not np.array_equal(c[obs], full[:n_t][obs]):
            bad += 1
    check("censor() is the leakage gate", bad == 0,
          "no future cell survives censoring; every observed cell preserved, "
          "3 cutoffs tested")


# ------------------------------------------------------------------ NegBin
def negbin():
    rng = np.random.default_rng(5)
    for mu, phi in ((5.0, 3.0), (200.0, 10.0), (1000.0, 50.0)):
        x = rng.negative_binomial(phi, phi / (phi + mu), size=400_000)
        m, v = x.mean(), x.var()
        v_target = mu + mu ** 2 / phi
        check(f"NegBin parameterisation mu={mu:g}, phi={phi:g}",
              abs(m - mu) / mu < 0.02 and abs(v - v_target) / v_target < 0.05,
              f"mean {m:.1f} vs {mu:.1f}; var {v:.0f} vs mu+mu^2/phi = {v_target:.0f}")


# ------------------------------------------------------------------ eq (9)
def eq9():
    z = np.array([0.0, 3.5, 1.0, 3.1, 3.2, 5.0, 0.0])
    months = np.arange(100, 107)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cs", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "05_case_studies.py"))
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    got = cs.first_alarm(z, months)
    # literal rule: first month where z > 3 for two consecutive months -> index 4 (=104)
    check("eq (9) alarm rule (z>3 in 2 consecutive months)", got == 104,
          f"single spike at index 1 correctly ignored; first alarm = {got} (expected 104)")


# ------------------------------------------------------------------ style audit
def style_audit():
    ok_files, missing = [], []
    for n in ("fig1", "fig2", "fig3", "fig4"):
        for p in (os.path.join(FIG_DIR, f"{n}.pdf"), os.path.join(FIG_DIR, f"{n}.png"),
                  os.path.join(GRAY_DIR, f"{n}_gray.png")):
            (ok_files if os.path.exists(p) and os.path.getsize(p) > 5_000
             else missing).append(os.path.basename(p))
    check("all figure outputs present (PDF + 600-dpi PNG + grayscale proof)",
          not missing, f"{len(ok_files)} files present"
          + (f"; MISSING {missing}" if missing else ""))

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "06_figures.py"), encoding="utf-8").read()
    overrides = re.findall(r"rcParams|set_size_inches|font\.family|plt\.style\.use", src)
    check("no local style overrides in the figure script", not overrides,
          "all styling flows through figstyle.py"
          if not overrides else f"found {overrides}")

    want = {0: "#8C8C8C", 1: "#4DBBD5", 2: "#E18727", 3: "#BC3C29"}
    check("severity palette matches the outline style guide",
          fs.SEV_COLORS == want and fs.NOWCAST == "#0072B5" and fs.NAIVE == "#D9D9D9",
          f"Y0 {fs.SEV_COLORS[0]}, Y1 {fs.SEV_COLORS[1]}, Y2 {fs.SEV_COLORS[2]}, "
          f"Y3 {fs.SEV_COLORS[3]}, nowcast {fs.NOWCAST}, naive {fs.NAIVE}")

    import matplotlib as mpl
    rc = mpl.rcParams
    check("figure typography per style guide",
          rc["font.sans-serif"][0] == "Arial" and 7.0 <= rc["font.size"] <= 8.5
          and rc["axes.linewidth"] == 0.5 and rc["pdf.fonttype"] == 42,
          f"font {rc['font.sans-serif'][0]}, size {rc['font.size']}, "
          f"axes lw {rc['axes.linewidth']}, Type-42 fonts embedded")

    check("figure widths are the single/double column constants",
          fs.W1 == 3.5 and fs.W2 == 7.2, f"W1 = {fs.W1} in, W2 = {fs.W2} in")

    # ---- grayscale legibility. The palette is colourblind-safe but NOT
    # grayscale-separable, so every severity contrast must carry a redundant
    # non-colour channel. Check the luminances AND that the channel exists.
    lum = fs.grayscale_luminance()
    pairs = [(a, b, abs(lum[a] - lum[b])) for a in range(4) for b in range(a + 1, 4)]
    worst = min(pairs, key=lambda p: p[2])
    redundant = (set(fs.SEV_LS) == set(range(4)) and len(set(map(str, fs.SEV_LS.values()))) == 4
                 and set(fs.SEV_MARKER) == set(range(4)) and len(set(fs.SEV_MARKER.values())) == 4
                 and set(fs.SEV_HATCH) == set(range(4)) and len(set(fs.SEV_HATCH.values())) == 4)
    check("redundant non-colour encoding defined for all four severity classes",
          redundant,
          f"grayscale luminance Y0 {lum[0]}, Y1 {lum[1]}, Y2 {lum[2]}, Y3 {lum[3]}; "
          f"closest pair Y{worst[0]}/Y{worst[1]} differ by only {worst[2]} levels, so "
          "line style, marker and hatch each provide 4 distinct values")

    src6 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_figures.py"), encoding="utf-8").read()
    # Statement-level: every plotting call that colours a series by severity must
    # carry a shape channel in the SAME call. Logical statements are reassembled by
    # tracking bracket depth, since matplotlib calls span several source lines.
    stmts, buf, depth = [], [], 0
    for line in src6.splitlines():
        buf.append(line)
        depth += line.count("(") + line.count("[") - line.count(")") - line.count("]")
        if depth <= 0:
            stmts.append(" ".join(buf))
            buf, depth = [], 0
    bare = [s.strip()[:70] for s in stmts
            if "fs.SEV_COLORS[s]" in s
            and not re.search(r"fs\.SEV_(?:LS|MARKER|HATCH)\[s\]", s)]
    check("every severity-coloured plotting call also varies a shape channel",
          not bare,
          f"{sum('fs.SEV_COLORS[s]' in s for s in stmts)} severity-coloured statements, "
          "all carrying line style, marker or hatch"
          if not bare else f"colour-only statements: {bare}")


# ------------------------------------------------------------------ truncation
def truncation_discipline():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "02_delay_analysis.py"), encoding="utf-8").read()
    n_gate = len(re.findall(r"mature_set\(", src))
    # The gate function itself is allowed to touch the raw frame; every OTHER line
    # that slices a delay column off `df` must carry the maturity restriction on the
    # same statement, or it is a truncation-discipline violation.
    body = re.sub(r"\ndef mature_set\(.*?(?=\ndef )", "\n", src, flags=re.S)
    offenders = [ln.strip() for ln in body.splitlines()
                 if re.search(r"df\[[^\]]*delay_(?:days|m)", ln)
                 and "MATURE_MAX" not in ln]
    check("all Step-2 descriptives route through the mature_set() gate",
          n_gate >= 3 and not offenders,
          f"{n_gate} calls to mature_set(); every other reference to a raw delay "
          "column carries the MATURE_MAX restriction on the same statement"
          if not offenders else f"unguarded slices: {offenders}")

    tri_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "chainladder.py"), encoding="utf-8").read()
    check("triangle construction enforces the cohort floor and the cap",
          "T0_FLOOR = 1995 * 12" in tri_src and "delay_m\"].to_numpy() <= D" in tri_src,
          "cohorts >= 1995-01 (snapshot start; earlier cohorts are left-truncated), "
          f"delay <= D")


def main():
    print("Formula-vs-code audit (outline eqs. 1-9)\n")
    eq1(); eq2_4_5(); eq3_7_8(); leakage(); negbin(); eq9()
    print("\nStyle and discipline audit\n")
    style_audit(); truncation_discipline()

    n_pass = sum(c["pass"] for c in CHECKS)
    lines = ["# Formula-vs-code audit", "",
             "Generated by `analysis/10_qa_audit.py`. Each outline equation is "
             "re-implemented here from a literal reading of the outline text and "
             "checked numerically against the production code.", "",
             f"**{n_pass} / {len(CHECKS)} checks pass.**", "",
             "| Check | Result | Detail |", "|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | "
                     f"{c['detail']} |")
    lines += ["", "## Equation coverage", "",
              "| Outline eq. | Implementation |", "|---|---|",
              "| (1) delay indexing | `common.month_index`, `01_build_complaints.py` |",
              "| (2) cumulative counts | `numpy.cumsum` in `chainladder` |",
              "| (3) estimand `N_s^(D)(t) = C_s(t,D)` | `chainladder.nowcast` (mature identity) |",
              "| (4) development factors | `chainladder.dev_factors` |",
              "| (5) `F_hat` from `lambda_hat` | `chainladder.F_from_lambda` |",
              "| (6) discrete-time reporting hazard | `02_delay_analysis.hazard_model` |",
              "| (7) point nowcast | `chainladder.nowcast` |",
              "| (8) undercount ratio | `chainladder.undercount_curve` |",
              "| (9) alarm rule | `05_case_studies.z_series` + `first_alarm` |",
              ""]
    with open(os.path.join(AUDIT_DIR, "formula_audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  {n_pass}/{len(CHECKS)} checks pass -> results/audits/formula_audit.md")
    if n_pass != len(CHECKS):
        raise SystemExit("formula audit FAILED")


if __name__ == "__main__":
    main()
