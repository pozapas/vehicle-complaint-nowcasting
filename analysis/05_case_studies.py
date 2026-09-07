"""Step 5  -  retrospective early-warning case studies (RQ3), eq (9).

Everything is run in PSEUDO-REAL-TIME: at each monthly cutoff T* the development
factors come from `chainladder.censor(tri, T*)`, and the monitored count for cohort
T* is what had actually arrived by T* (delay 0). The alarm rule is
    z_j(t) = (N_j(t) - mu_j(t)) / sigma_j(t),  alarm when z > 3 in two consecutive months.

Three arms are compared, not two:
  naive          current partial count vs a baseline of prior same-month counts as
                 they stand at T* (what a dashboard built on raw counts actually does);
  naive_matched  same current count vs a MATURITY-MATCHED baseline -- prior same-month
                 counts observed at their own delay 0. This is the strong version of
                 the naive comparator and guards against a strawman;
  nowcast        delay-corrected current count vs a delay-corrected baseline.

Stream-level triangles are far too thin to support their own chain ladder, so the
severity-pooled F_hat for severe (Y>=1) consumer complaints is used as the correction
factor -- exactly the deployable "lookup table refreshed monthly" of outline §6.

Outputs
  results/audits/case_studies.json
  results/tables/case_z_traces.csv      z trajectories for Fig 4c
  results/tables/detection_gains.csv    stream-wide Delta distribution
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainladder as cl
from common import (AUDIT_DIR, D_CAP, PARQUET, T_CUTOFF, TAB_DIR, W_DEFAULT,
                    p_to_str, qa_log, str_to_p, write_json)

T0 = cl.T0_FLOOR
Z_THRESH = 3.0
N_CONSEC = 2
N_BASE_YEARS = 5
MIN_SURGE = 5           # complaints/month at surge below which a case is dropped
SCAN_FIRST = str_to_p("2005-01")
SCAN_LAST = str_to_p("2023-06")

# Primary cases, fixed a priori (outline §4.5 / Step 5)  -  chosen before seeing results.
CASES = [
    {"name": "Hyundai/Kia engine fires",
     "makes": ["HYUNDAI", "KIA"],
     "families": ["ENGINE", "ENGINE AND ENGINE COOLING"],
     "window": ("2017-01", "2020-12")},
    {"name": "Chevrolet Bolt battery fires",
     "makes": ["CHEVROLET"],
     "families": ["ELECTRICAL SYSTEM"],
     "window": ("2019-01", "2022-12")},
    {"name": "ARC inflator (all makes, air bags)",
     "makes": None,
     "families": ["AIR BAGS"],
     "window": ("2021-01", "2024-12")},
]


def build_stream_triangle(df, mask):
    return cl.build_triangle(df, T_CUTOFF, D_CAP, t0=T0, mask=mask)


def pooled_F_by_cutoff(sev_tri, cutoffs):
    """F_hat(severe) estimated at each pseudo-cutoff from the censored triangle."""
    out = {}
    for Ts in cutoffs:
        tri_c = cl.censor(sev_tri, Ts, D_CAP, t0=T0)
        out[Ts] = cl.fit_F(tri_c, Ts, D_CAP, W_DEFAULT, t0=T0)[1]
    return out


def z_series(tri, Fmap, cutoffs):
    """Pseudo-real-time z for the three arms, one value per cutoff."""
    C = np.cumsum(tri, axis=1)
    n_t = tri.shape[0]
    rows = []
    for Ts in cutoffs:
        i = Ts - T0
        if i < N_BASE_YEARS * 12 or i >= n_t:
            continue
        F = Fmap[Ts]
        cur_obs = C[i, 0]                       # arrived by end of the incident month
        cur_now = cur_obs / max(F[0], 1e-9)

        base_idx = [i - 12 * k for k in range(1, N_BASE_YEARS + 1)]
        h = [min(Ts - (T0 + j), D_CAP) for j in base_idx]
        base_obs = np.array([C[j, hh] for j, hh in zip(base_idx, h)])       # as of T*
        base_now = np.array([C[j, hh] / max(F[hh], 1e-9)
                             for j, hh in zip(base_idx, h)])
        base_matched = np.array([C[j, 0] for j in base_idx])                # own delay 0

        rec = {"Tstar": Ts, "month": p_to_str(Ts), "obs": cur_obs, "nowcast": cur_now}
        for arm, cur, base in (("naive", cur_obs, base_obs),
                               ("naive_matched", cur_obs, base_matched),
                               ("nowcast", cur_now, base_now)):
            mu = float(base.mean())
            sd = max(float(base.std(ddof=1)) if len(base) > 1 else 0.0,
                     float(np.sqrt(max(mu, 1.0))))       # Poisson floor
            if arm == "nowcast":
                # The nowcast arm divides the current partial count by F_hat(0),
                # which amplifies its sampling variance. `sd` above is the
                # month-to-month dispersion of a FULLY DEVELOPED month; the current
                # statistic N_hat = C/F_hat(0) carries the extra arrival variance of
                # the 1 - F_hat(0) share that has not yet been observed. Under a
                # Poisson arrival null, Var(C/F) = mu/F, so the additional term is
                # mu * (1/F - 1); adding it to the developed dispersion reduces to
                # sd = sqrt(mu/F) exactly when the baseline is Poisson, and to
                # sd = sd_developed when F = 1. Without this term the nowcast arm's
                # z is inflated by roughly 1/sqrt(F) relative to the maturity-matched
                # arm, which would make any detection gain an artifact of the
                # division rather than a property of the signal.
                f0 = float(max(F[0], 1e-9))
                sd = float(np.sqrt(sd ** 2 + mu * (1.0 / f0 - 1.0)))
            rec[f"mu_{arm}"] = mu
            rec[f"sd_{arm}"] = sd
            rec[f"z_{arm}"] = (cur - mu) / sd
        rows.append(rec)
    return pd.DataFrame(rows)


def first_alarm(z, months, thresh=Z_THRESH):
    """First month at which z > threshold for N_CONSEC consecutive months."""
    z = np.asarray(z, dtype=float)
    run = 0
    for k in range(len(z)):
        run = run + 1 if z[k] > thresh else 0
        if run >= N_CONSEC:
            return int(months[k])
    return None


def match_threshold(traces, target_n, lo=0.0, hi=Z_THRESH, tol=1e-3):
    """Lowest naive threshold whose alarming-stream count reaches `target_n`.

    Sensitivity is trivially bought by lowering the bar, so comparing arms that
    alarm at different rates says nothing about detection. This calibrates the
    naive arm to the SAME number of alarming streams as the corrected arm, so the
    two are compared at matched specificity and only the timing can differ.
    """
    def n_alarm(th):
        return sum(first_alarm(z, m, th) is not None for z, m in traces)

    if n_alarm(hi) >= target_n:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if n_alarm(mid) >= target_n:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    df = pd.read_parquet(PARQUET)
    cons = df[df["consumer"]].copy()
    sev_mask = (cons["severity"].to_numpy() >= 1)
    sev_tri = cl.build_triangle(cons, T_CUTOFF, D_CAP, t0=T0, mask=sev_mask)

    cutoffs = list(range(SCAN_FIRST, min(SCAN_LAST, T_CUTOFF) + 1))
    print(f"  estimating pooled F_hat at {len(cutoffs)} pseudo-cutoffs ...", flush=True)
    Fmap = pooled_F_by_cutoff(sev_tri, cutoffs)

    make = cons["make"].astype(str).to_numpy()
    fam = cons["comp_family"].astype(str).to_numpy()

    out = {"rule": {"z_threshold": Z_THRESH, "consecutive_months": N_CONSEC,
                    "baseline": f"same calendar month, prior {N_BASE_YEARS} years",
                    "sigma_floor": "Poisson sd sqrt(mu)",
                    "nowcast_arm_variance": "sd = sqrt(sd_developed^2 + mu*(1/F_hat(0) - 1)); "
                                            "propagates the arrival variance of the "
                                            "not-yet-observed share into the alarm "
                                            "denominator, so the correction cannot buy "
                                            "sensitivity through the division alone",
                    "correction": "severity-pooled F_hat for severe (Y>=1) consumer "
                                  "complaints, re-estimated at each cutoff"},
           "arms": ["naive", "naive_matched", "nowcast"],
           "primary_cases": []}

    traces = []
    for case in CASES:
        m = sev_mask & np.isin(fam, case["families"])
        if case["makes"]:
            m = m & np.isin(make, case["makes"])
        tri = build_stream_triangle(cons, m)
        w0, w1 = str_to_p(case["window"][0]), str_to_p(case["window"][1])
        wc = [t for t in cutoffs if w0 <= t <= w1]
        if not wc:
            out["primary_cases"].append({"name": case["name"],
                                         "status": "window outside scan range"})
            continue
        z = z_series(tri, Fmap, wc)
        peak = float(np.cumsum(tri, axis=1)[:, D_CAP][[t - T0 for t in wc]].max())

        rec = {"name": case["name"], "makes": case["makes"],
               "families": case["families"],
               "window": list(case["window"]),
               "peak_monthly_severe_complete": peak,
               "total_severe_in_window": float(
                   np.cumsum(tri, axis=1)[:, D_CAP][[t - T0 for t in wc]].sum())}
        if peak < MIN_SURGE:
            rec["status"] = (f"DROPPED  -  stream too thin (peak {peak:.0f} < "
                             f"{MIN_SURGE} severe complaints/month at surge)")
            out["primary_cases"].append(rec)
            continue

        rec["status"] = "evaluated"
        taus = {}
        for arm in ("naive", "naive_matched", "nowcast"):
            t_a = first_alarm(z[f"z_{arm}"], z["Tstar"])
            taus[arm] = p_to_str(t_a) if t_a is not None else None
            rec[f"tau_{arm}"] = taus[arm]
            rec[f"tau_{arm}_p"] = t_a
        for comp in ("naive", "naive_matched"):
            a, b = rec[f"tau_{comp}_p"], rec["tau_nowcast_p"]
            rec[f"delta_vs_{comp}_months"] = (a - b) if (a is not None and b is not None) \
                else ("nowcast only" if b is not None else
                      ("comparator only" if a is not None else "neither alarms"))
        rec["max_z"] = {a: float(z[f"z_{a}"].max()) for a in
                        ("naive", "naive_matched", "nowcast")}
        out["primary_cases"].append(rec)

        z2 = z.copy()
        z2.insert(0, "case", case["name"])
        traces.append(z2)

    if traces:
        pd.concat(traces, ignore_index=True).to_csv(
            os.path.join(TAB_DIR, "case_z_traces.csv"), index=False)

    # ------------------------------------------------------------ stream-wide sweep
    print("  stream-wide sweep ...", flush=True)
    sub = cons[sev_mask]
    pair = sub.groupby([sub["make"].astype(str), sub["comp_family"].astype(str)],
                       observed=True).size()
    cand = pair[pair >= 60].index.tolist()      # cheap prefilter before the 24-mo test
    print(f"    {len(cand)} candidate make x component-family streams", flush=True)

    gains, kept = [], 0
    naive_traces, matched_traces = [], []
    for mk, fm in cand:
        m = sev_mask & (make == mk) & (fam == fm)
        tri = build_stream_triangle(cons, m)
        tot = np.cumsum(tri, axis=1)[:, D_CAP]
        roll = np.convolve(tot, np.ones(24), mode="valid")
        if roll.max() < 30:                      # outline: >=30 severe in any 24-mo window
            continue
        kept += 1
        z = z_series(tri, Fmap, cutoffs)
        if len(z) == 0:
            continue
        r = {"make": mk, "comp_family": fm, "peak_24mo_severe": float(roll.max())}
        naive_traces.append((np.asarray(z["z_naive"], dtype=float),
                             np.asarray(z["Tstar"])))
        matched_traces.append((np.asarray(z["z_naive_matched"], dtype=float),
                               np.asarray(z["Tstar"])))
        for arm in ("naive", "naive_matched", "nowcast"):
            t_a = first_alarm(z[f"z_{arm}"], z["Tstar"])
            r[f"tau_{arm}"] = t_a
            r[f"tau_{arm}_str"] = p_to_str(t_a) if t_a is not None else None
        r["delta_vs_naive"] = (r["tau_naive"] - r["tau_nowcast"]) \
            if (r["tau_naive"] is not None and r["tau_nowcast"] is not None) else np.nan
        r["delta_vs_naive_matched"] = (r["tau_naive_matched"] - r["tau_nowcast"]) \
            if (r["tau_naive_matched"] is not None and r["tau_nowcast"] is not None) else np.nan
        gains.append(r)

    gdf = pd.DataFrame(gains)
    # NOTE: written to disk after the specificity-matched columns are added below,
    # so that Fig 4(d) can read the per-stream matched-specificity gains.

    def summarise(col):
        d = gdf[col].dropna()
        if len(d) == 0:
            return {"n": 0}
        return {"n": int(len(d)), "mean": float(d.mean()), "median": float(d.median()),
                "share_gain_ge_1": float((d >= 1).mean()),
                "share_zero": float((d == 0).mean()),
                "share_negative": float((d < 0).mean()),
                "p25": float(d.quantile(.25)), "p75": float(d.quantile(.75)),
                "max": float(d.max()), "min": float(d.min())}

    # --- specificity-matched comparison -------------------------------------
    # Alarm counts are not comparable across arms at a common threshold, because
    # the corrected arm alarms far more often. Recalibrate each naive arm to the
    # SAME number of alarming streams as the nowcast arm, then compare timing.
    n_now = int(gdf["tau_nowcast"].notna().sum())
    spec = {}
    for arm, traces in (("naive", naive_traces), ("naive_matched", matched_traces)):
        th = match_threshold(traces, n_now)
        taus = [first_alarm(zz, mm, th) for zz, mm in traces]
        gdf[f"tau_{arm}_spec"] = taus
        gdf[f"delta_vs_{arm}_spec"] = gdf[f"tau_{arm}_spec"] - gdf["tau_nowcast"]
        d = gdf[f"delta_vs_{arm}_spec"].dropna()
        spec[arm] = {
            "threshold": float(th),
            "n_alarming_streams": int(sum(t is not None for t in taus)),
            "n_both": int(len(d)),
            "median_gain_months": float(d.median()) if len(d) else None,
            "mean_gain_months": float(d.mean()) if len(d) else None,
            "p25": float(d.quantile(.25)) if len(d) else None,
            "p75": float(d.quantile(.75)) if len(d) else None,
            "share_gain_ge_1": float((d >= 1).mean()) if len(d) else None,
            "share_zero": float((d == 0).mean()) if len(d) else None,
            "share_negative": float((d < 0).mean()) if len(d) else None,
        }
    spec["n_alarming_streams_nowcast"] = n_now
    gdf.to_csv(os.path.join(TAB_DIR, "detection_gains.csv"), index=False)

    # --- gains restricted to alarms within 12 months of each other ----------
    # The unrestricted metric differences first-EVER threshold crossings over a
    # 30-year record, so two unrelated episodes a decade apart count as a "gain".
    def summarise_windowed(col, w=12):
        d = gdf[col].dropna()
        d = d[d.abs() <= w]
        if len(d) == 0:
            return {"n": 0}
        return {"n": int(len(d)), "mean": float(d.mean()), "median": float(d.median()),
                "share_gain_ge_1": float((d >= 1).mean()),
                "share_zero": float((d == 0).mean()),
                "share_negative": float((d < 0).mean()),
                "p25": float(d.quantile(.25)), "p75": float(d.quantile(.75)),
                "max": float(d.max()), "min": float(d.min())}

    out["stream_wide"] = {
        "n_streams_screened": len(cand),
        "n_streams_evaluated": kept,
        "alarm_rate_nowcast": float(gdf["tau_nowcast"].notna().mean()),
        "alarm_rate_naive": float(gdf["tau_naive"].notna().mean()),
        "alarm_rate_naive_matched": float(gdf["tau_naive_matched"].notna().mean()),
        "nowcast_only_alarms": int((gdf["tau_naive"].isna()
                                    & gdf["tau_nowcast"].notna()).sum()),
        "naive_only_alarms": int((gdf["tau_naive"].notna()
                                  & gdf["tau_nowcast"].isna()).sum()),
        "neither_alarms": int((gdf["tau_naive"].isna() & gdf["tau_nowcast"].isna()).sum()),
        "delta_vs_naive": summarise("delta_vs_naive"),
        "delta_vs_naive_matched": summarise("delta_vs_naive_matched"),
        "delta_vs_naive_within12": summarise_windowed("delta_vs_naive"),
        "delta_vs_naive_matched_within12": summarise_windowed("delta_vs_naive_matched"),
        "specificity_matched": spec,
    }

    # ------------------------------------------------------------ Gate 5
    ev = [c for c in out["primary_cases"] if c.get("status") == "evaluated"]
    deltas = [c.get("delta_vs_naive_months") for c in ev]
    num = [d for d in deltas if isinstance(d, (int, float))]
    g5 = any(d >= 1 for d in num) or any(d == "nowcast only" for d in deltas)
    out["gate5"] = {"pass": bool(g5),
                    "primary_deltas_vs_naive": {c["name"]: c.get("delta_vs_naive_months")
                                                for c in ev},
                    "primary_deltas_vs_naive_matched": {
                        c["name"]: c.get("delta_vs_naive_matched_months") for c in ev},
                    "honest_negative": not bool(g5)}
    qa_log("5", "Gate 5  -  at least one primary case shows detection gain >= 1 month", g5,
           "; ".join(f"{c['name']}: {c.get('delta_vs_naive_months')}" for c in ev)
           or "no case evaluated")

    write_json(os.path.join(AUDIT_DIR, "case_studies.json"), out)

    print("\n  primary cases:")
    for c in out["primary_cases"]:
        if c.get("status") != "evaluated":
            print(f"    {c['name']}: {c.get('status')}")
            continue
        print(f"    {c['name']}")
        print(f"      tau naive {c['tau_naive']}  matched {c['tau_naive_matched']}  "
              f"nowcast {c['tau_nowcast']}")
        print(f"      Delta vs naive: {c['delta_vs_naive_months']}   "
              f"vs matched: {c['delta_vs_naive_matched_months']}")
    sw = out["stream_wide"]
    print(f"\n  stream-wide: {sw['n_streams_evaluated']} streams evaluated")
    print(f"    Delta vs naive: {sw['delta_vs_naive']}")
    print(f"    Delta vs naive_matched: {sw['delta_vs_naive_matched']}")
    print(f"    nowcast-only alarms: {sw['nowcast_only_alarms']}, "
          f"naive-only: {sw['naive_only_alarms']}, neither: {sw['neither_alarms']}")
    print(f"    alarm rates: nowcast {sw['alarm_rate_nowcast']:.3f}  "
          f"naive {sw['alarm_rate_naive']:.3f}  "
          f"matched {sw['alarm_rate_naive_matched']:.3f}")
    print(f"    within-12mo gains vs naive:   {sw['delta_vs_naive_within12']}")
    print(f"    within-12mo gains vs matched: {sw['delta_vs_naive_matched_within12']}")
    print(f"    SPECIFICITY-MATCHED: {sw['specificity_matched']}")
    print("\n  GATE 5 PASSED" if g5 else
          "\n  GATE 5  -  honest negative recorded (no manufactured gain)")


if __name__ == "__main__":
    main()
