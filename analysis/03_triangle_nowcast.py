"""Step 3  -  reporting triangles, chain ladder, undercount curves, unit tests.

The three asserts required by the outline are MANDATORY and run before any real
data is touched (the synthetic one validates eqs (4)-(5)-(7) end-to-end, so an
off-by-one cannot silently propagate downstream).

Outputs
  results/tables/undercount_curves.csv   F_hat_s(d) per severity stream and window
  results/tables/triangle_all.csv        cohort x delay counts (Fig 3a heatmap)
  results/tables/nowcast_points.csv      point nowcasts for recent cohorts
  results/audits/triangle_audit.json
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainladder as cl
from common import (AUDIT_DIR, D_CAP, MATURE_MAX, PARQUET, SEED, SEV_LABELS, T_CUTOFF,
                    TAB_DIR, W_DEFAULT, p_to_str, qa_log, write_json)

T0 = cl.T0_FLOOR
STREAMS = ("all", "Y0", "Y1", "Y2", "Y3", "severe")


def stream_mask(df, name):
    sev = df["severity"].to_numpy()
    if name == "all":
        return None, None
    if name == "severe":
        return None, sev >= 1
    return int(name[1]), None


# ------------------------------------------------------------------ unit tests
def test_synthetic_recovery():
    """(i) On synthetic multinomial data with known p(d), the estimator recovers F."""
    rng = np.random.default_rng(7)
    D = 12
    T = 300
    p = np.array([0.30, 0.22, 0.14, 0.09, 0.06, 0.05, 0.04, 0.03, 0.025,
                  0.02, 0.015, 0.01, 0.01])
    p = p / p.sum()
    F_true = np.cumsum(p)

    n_t = T + 1
    N = rng.integers(3000, 6000, size=n_t)
    tri = np.zeros((n_t, D + 1))
    for i in range(n_t):
        tri[i] = rng.multinomial(N[i], p)
    # right-truncate exactly as the real snapshot is: cell (t,d) observed iff t+d <= T
    tri_obs = cl.censor(tri, T, D, t0=0)

    lam, F_hat = cl.fit_F(tri_obs, T, D, W=10_000, t0=0)
    err = np.max(np.abs(F_hat - F_true))
    assert err < 0.01, f"synthetic recovery failed: max|F_hat - F| = {err:.4f}"

    # and the nowcast must recover the (unobserved) true cohort totals in expectation
    _, _, N_hat, h = cl.nowcast(tri_obs, F_hat, T, D, t0=0)
    recent = h < D
    rel = np.abs(N_hat[recent] - N[recent]) / N[recent]
    assert np.median(rel) < 0.05, f"synthetic nowcast median rel err {np.median(rel):.4f}"
    return {"max_abs_F_error": float(err),
            "median_rel_nowcast_error_recent": float(np.median(rel))}


def test_mature_identity(tri, F, T, D):
    """(ii) For fully mature cohorts, N_hat(t) == C(t, D) exactly."""
    t, C_obs, N_hat, h = cl.nowcast(tri, F, T, D, t0=T0)
    mature = t <= (T - D)
    C_full = np.cumsum(tri, axis=1)[:, D]
    diff = np.max(np.abs(N_hat[mature] - C_full[mature]))
    assert diff < 1e-9, f"mature identity failed: max |N_hat - C(t,D)| = {diff}"
    return {"max_abs_diff": float(diff), "n_mature_cohorts": int(mature.sum())}


def test_F_monotone(F, D):
    """(iii) F_hat monotone nondecreasing in d, with F_hat(D) = 1."""
    assert np.all(np.diff(F) >= -1e-12), "F_hat not monotone in d"
    assert abs(F[D] - 1.0) < 1e-12, f"F_hat(D) = {F[D]!r}, expected 1"
    assert np.all(F > 0), "F_hat has non-positive entries"
    return {"F0": float(F[0]), "FD": float(F[D])}


# ------------------------------------------------------------------ main
def main():
    audit = {"T": p_to_str(T_CUTOFF), "D": D_CAP, "W_default": W_DEFAULT,
             "t0": p_to_str(T0), "streams": list(STREAMS)}

    print("  unit test (i) synthetic multinomial recovery ...", flush=True)
    audit["test_synthetic"] = test_synthetic_recovery()
    print(f"    max|F_hat - F| = {audit['test_synthetic']['max_abs_F_error']:.5f}  OK")

    df = pd.read_parquet(PARQUET)
    cons = df[df["consumer"]].copy()

    zero_logs = []
    tris, Fs, lams = {}, {}, {}
    for name in STREAMS:
        sev, msk = stream_mask(cons, name)
        tris[name] = cl.build_triangle(cons, T_CUTOFF, D_CAP, t0=T0, severity=sev, mask=msk)
        log = []
        lams[name], Fs[name] = cl.fit_F(tris[name], T_CUTOFF, D_CAP, W_DEFAULT, t0=T0, log=log)
        if log:
            zero_logs.append({"stream": name, "events": log})
    audit["zero_denominator_events"] = zero_logs

    print("  unit test (ii) mature-cohort identity N_hat(t) == C(t,D) ...", flush=True)
    audit["test_mature_identity"] = {
        n: test_mature_identity(tris[n], Fs[n], T_CUTOFF, D_CAP) for n in STREAMS}
    print("    OK")
    print("  unit test (iii) F_hat monotone with F_hat(D)=1 ...", flush=True)
    audit["test_F_monotone"] = {n: test_F_monotone(Fs[n], D_CAP) for n in STREAMS}
    print("    OK")

    # ---------------------------------------------------- undercount curves, eq (8)
    rows = []
    for W in (W_DEFAULT, 36, 10_000):
        wname = "all-history" if W == 10_000 else str(W)
        for name in STREAMS:
            sev, msk = stream_mask(cons, name)
            lam, F = cl.fit_F(tris[name], T_CUTOFF, D_CAP, W, t0=T0)
            for d in range(D_CAP + 1):
                rows.append({"stream": name, "W": wname, "d": d, "F": F[d],
                             "lam": lam[d] if d < D_CAP else np.nan})
    uc = pd.DataFrame(rows)
    uc.to_csv(os.path.join(TAB_DIR, "undercount_curves.csv"), index=False)

    main_uc = uc[uc["W"] == str(W_DEFAULT)]
    audit["undercount_ratios"] = {
        s: {f"F({h})": float(main_uc[(main_uc.stream == s) & (main_uc.d == h)]["F"].iloc[0])
            for h in (0, 1, 3, 6, 12, 24, 36)} for s in STREAMS}

    # ---------------------------------------------------- triangle export (Fig 3a)
    tri_all = tris["all"]
    n_t = tri_all.shape[0]
    tdf = pd.DataFrame(tri_all, columns=[f"d{d}" for d in range(D_CAP + 1)])
    tdf.insert(0, "cohort", [p_to_str(T0 + i) for i in range(n_t)])
    tdf.insert(1, "cohort_p", np.arange(n_t) + T0)
    tdf.insert(2, "observed_cells", np.minimum(T_CUTOFF - (np.arange(n_t) + T0), D_CAP) + 1)
    tdf.to_csv(os.path.join(TAB_DIR, "triangle_all.csv"), index=False)

    # ---------------------------------------------------- point nowcasts, eq (7)
    nrows = []
    for name in STREAMS:
        t, C_obs, N_hat, h = cl.nowcast(tris[name], Fs[name], T_CUTOFF, D_CAP, t0=T0)
        for i in range(len(t)):
            if t[i] < T_CUTOFF - 47:      # keep the last 48 cohorts for figures
                continue
            nrows.append({"stream": name, "cohort": p_to_str(t[i]), "cohort_p": int(t[i]),
                          "h": int(h[i]), "C_obs": float(C_obs[i]),
                          "F_hat_h": float(Fs[name][h[i]]), "N_hat": float(N_hat[i]),
                          "implied_missing": float(N_hat[i] - C_obs[i]),
                          "mature": bool(t[i] <= MATURE_MAX)})
    pd.DataFrame(nrows).to_csv(os.path.join(TAB_DIR, "nowcast_points.csv"), index=False)

    # ---------------------------------------------------- Gate 3
    f3_y3 = audit["undercount_ratios"]["Y3"]["F(3)"]
    f3_y0 = audit["undercount_ratios"]["Y0"]["F(3)"]
    g3 = f3_y3 < f3_y0
    qa_log("3", "Gate 3  -  unit tests (i)-(iii) all pass", True,
           f"synthetic max|dF| = {audit['test_synthetic']['max_abs_F_error']:.5f}; "
           "mature identity exact; F monotone with F(D)=1")
    qa_log("3", "Gate 3  -  F_hat_Y3(3) < F_hat_Y0(3)", g3,
           f"Y3 {f3_y3:.3f} vs Y0 {f3_y0:.3f}")
    audit["gate3"] = {"F_Y3_3": f3_y3, "F_Y0_3": f3_y0, "pass": bool(g3)}

    write_json(os.path.join(AUDIT_DIR, "triangle_audit.json"), audit)

    print("\n  undercount ratios F_hat(h) (consumer, W=60):")
    print(f"    {'stream':<8}" + "".join(f"{'h=' + str(h):>9}" for h in (1, 3, 6, 12, 24)))
    for s in STREAMS:
        r = audit["undercount_ratios"][s]
        print(f"    {s:<8}" + "".join(f"{r[f'F({h})']:>9.3f}" for h in (1, 3, 6, 12, 24)))
    if not g3:
        raise SystemExit("GATE 3 FAILED")
    print("  GATE 3 PASSED")


if __name__ == "__main__":
    main()
