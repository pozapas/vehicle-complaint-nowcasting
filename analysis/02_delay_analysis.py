"""Step 2  -  reporting-delay descriptives and the discrete-time hazard model (RQ1).

EVERYTHING here runs on MATURE cohorts only (incident cohort <= 2023-06 = T - D),
consumer channels only. Recent cohorts are right-truncated and would bias every
delay statistic short; the single `mature_set()` gate below is the only entry point.

Outputs
  results/audits/delay_stats.json
  results/tables/ecdf_by_severity.csv, ecdf_by_channel.csv, ecdf_by_era.csv
  results/tables/delay_quantiles.csv, delay_drift.csv, hazard_model.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUDIT_DIR, D_CAP, ERAS, MATURE_MAX, PARQUET, SEED, SEV_LABELS,
                    TAB_DIR, qa_log, write_json)

DAY_CAP = 3650   # 10 years: day-level descriptives only (outline Step 2)
N_BOOT_MED = 1000
TOP_COMPFAM = 8


def load():
    df = pd.read_parquet(PARQUET)
    return df


def mature_set(df, consumer_only=True, day_level=False):
    """THE truncation gate. Mature cohorts only; nothing else may bypass this."""
    m = df["valid_delay"] & (df["cohort_p"] <= MATURE_MAX) & (df["cohort_p"] >= 1995 * 12)
    if consumer_only:
        m &= df["consumer"]
    m &= (df["delay_days"] <= DAY_CAP) if day_level else (df["delay_m"] <= D_CAP)
    out = df.loc[m].copy()
    out["cohort_year"] = (out["cohort_p"] // 12).astype(int)
    out["era"] = pd.cut(out["cohort_year"],
                        bins=[1994, 2004, 2014, 2023],
                        labels=[e[0] for e in ERAS]).astype(str)
    out["delay_m"] = out["delay_m"].astype(int)
    return out


def quantiles(x, qs=(0.25, 0.5, 0.75, 0.9, 0.95)):
    x = np.asarray(x, dtype="float64")
    if len(x) == 0:
        return {f"p{int(q * 100)}": None for q in qs}
    v = np.quantile(x, qs)
    return {f"p{int(q * 100)}": float(v[i]) for i, q in enumerate(qs)}


def ecdf_months(sub):
    """Empirical F(d), d = 0..D, on the delay-capped mature set."""
    cnt = np.bincount(sub["delay_m"].to_numpy(), minlength=D_CAP + 1)[: D_CAP + 1]
    n = cnt.sum()
    return (np.cumsum(cnt) / n if n else np.full(D_CAP + 1, np.nan)), int(n)


def boot_median_diff(a, b, rng, B=N_BOOT_MED):
    """Bootstrap CI for median(a) - median(b)."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    d = np.empty(B)
    for i in range(B):
        d[i] = (np.median(rng.choice(a, len(a), replace=True))
                - np.median(rng.choice(b, len(b), replace=True)))
    return {"point": float(np.median(a) - np.median(b)),
            "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5))}


# ------------------------------------------------------------------ hazard model
def hazard_model(mat, out_csv):
    """Discrete-time logistic reporting hazard, eq (6), on person-period counts.

    h(d) = Pr(D = d | D >= d, x) = 1 - exp(-exp(alpha_d + x'beta)).
    Fitted on aggregated (delay-bin x severity x era x channel x component-family)
    cells with binomial (events, at-risk - events) response.
    """
    top = mat["comp_family"].value_counts().head(TOP_COMPFAM).index.tolist()
    d = mat.copy()
    d["compfam"] = np.where(d["comp_family"].isin(top), d["comp_family"].astype(str), "OTHER")
    d["channel"] = d["cmpl_type"].astype(str)

    keys = ["severity", "era", "channel", "compfam"]
    tab = (d.groupby(keys + ["delay_m"], observed=True).size()
             .unstack("delay_m", fill_value=0)
             .reindex(columns=range(D_CAP + 1), fill_value=0))
    ev = tab.to_numpy(dtype="int64")
    at_risk = ev[:, ::-1].cumsum(axis=1)[:, ::-1]     # #still-unreported entering delay u

    idx = tab.index.to_frame(index=False)
    G, U = ev.shape
    long = pd.DataFrame({
        "severity": np.repeat(idx["severity"].to_numpy(), U),
        "era": np.repeat(idx["era"].to_numpy(), U),
        "channel": np.repeat(idx["channel"].to_numpy(), U),
        "compfam": np.repeat(idx["compfam"].to_numpy(), U),
        "d": np.tile(np.arange(U), G),
        "events": ev.ravel(),
        "at_risk": at_risk.ravel(),
    })
    long = long[long["at_risk"] > 0]

    # delay baseline: categorical 0..12, then binned 13-18 / 19-24 / 25-36
    dd = long["d"].to_numpy()
    dbin = np.where(dd <= 12, dd.astype(object),
                    np.where(dd <= 18, "13-18", np.where(dd <= 24, "19-24", "25-36")))
    long["dbin"] = [str(x) for x in dbin]
    long = (long.groupby(["dbin", "severity", "era", "channel", "compfam"], observed=True)
                [["events", "at_risk"]].sum().reset_index())

    # explicit reference levels (largest / baseline categories)
    ref = {"dbin": "0", "severity": 0, "era": "2015-2023",
           "channel": long.groupby("channel")["at_risk"].sum().idxmax(),
           "compfam": long.groupby("compfam")["at_risk"].sum().idxmax()}
    parts = []
    for col, r in ref.items():
        dum = pd.get_dummies(long[col].astype(str), prefix=col, dtype=float)
        dum = dum.drop(columns=[f"{col}_{r}"])
        parts.append(dum)
    X = pd.concat([pd.Series(1.0, index=long.index, name="const")] + parts, axis=1)

    endog = np.column_stack([long["events"].to_numpy(),
                             (long["at_risk"] - long["events"]).to_numpy()])
    # Complementary log-log link so that exp(beta) is a hazard ratio rather than an
    # odds ratio; the baseline reporting hazard is ~0.46 at d=0, far from the
    # small-hazard regime in which the two coincide.
    fit = sm.GLM(endog, X.to_numpy(),
                 family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()

    ci = fit.conf_int()
    res = pd.DataFrame({
        "term": X.columns,
        "beta": fit.params,
        "se": fit.bse,
        "hr": np.exp(fit.params),
        "hr_lo95": np.exp(ci[:, 0]),
        "hr_hi95": np.exp(ci[:, 1]),
        "p": fit.pvalues,
    })
    res["reference"] = res["term"].map(
        lambda t: ref.get(t.split("_")[0], "") if "_" in t else "")
    res.to_csv(out_csv, index=False)

    sev_rows = res[res["term"].str.startswith("severity_")]
    return {
        "n_cells": int(len(long)),
        "n_person_periods": int(long["at_risk"].sum()),
        "references": {k: str(v) for k, v in ref.items()},
        "pseudo_r2_dev": float(1 - fit.deviance / fit.null_deviance),
        "severity_hazard_ratios": {
            r["term"].replace("severity_", "Y"): {
                "hr": float(r["hr"]), "lo95": float(r["hr_lo95"]),
                "hi95": float(r["hr_hi95"]), "p": float(r["p"]),
            } for _, r in sev_rows.iterrows()
        },
        "csv": os.path.basename(out_csv),
    }


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    mat_d = mature_set(df, day_level=True)     # day-level descriptives (<= 3650 d)
    mat_m = mature_set(df, day_level=False)    # month-level, delay-capped at D

    stats_out = {
        "definition": {
            "T": "2026-06", "D_months": D_CAP,
            "mature_rule": "incident cohort <= 2023-06 (= T - D)",
            "cohort_floor": "1995-01 (snapshot starts 1995-01; earlier cohorts left-truncated)",
            "channels": "consumer only (VOQ, EVOQ, IVOQ, MAVQ)",
            "day_level_cap_days": DAY_CAP,
        },
        "n_mature_day_level": int(len(mat_d)),
        "n_mature_month_level": int(len(mat_m)),
    }

    # ---------------------------------------------------- day-level quantiles
    rows = []
    overall = quantiles(mat_d["delay_days"])
    rows.append({"group": "overall", "level": "all", "n": len(mat_d), **overall,
                 "mean": float(mat_d["delay_days"].mean())})
    for s in sorted(mat_d["severity"].unique()):
        sub = mat_d[mat_d["severity"] == s]
        rows.append({"group": "severity", "level": SEV_LABELS[s], "n": len(sub),
                     **quantiles(sub["delay_days"]),
                     "mean": float(sub["delay_days"].mean())})
    for ch, sub in mat_d.groupby(mat_d["cmpl_type"].astype(str), observed=True):
        if len(sub) < 100:
            continue
        rows.append({"group": "channel", "level": ch, "n": len(sub),
                     **quantiles(sub["delay_days"]), "mean": float(sub["delay_days"].mean())})
    for era, sub in mat_d.groupby("era", observed=True):
        rows.append({"group": "era", "level": era, "n": len(sub),
                     **quantiles(sub["delay_days"]), "mean": float(sub["delay_days"].mean())})
    qdf = pd.DataFrame(rows)
    qdf.to_csv(os.path.join(TAB_DIR, "delay_quantiles.csv"), index=False)
    stats_out["delay_quantiles_days"] = qdf.to_dict(orient="records")

    # ---------------------------------------------------- month-level ECDFs
    ec_rows = []
    for s in sorted(mat_m["severity"].unique()):
        F, n = ecdf_months(mat_m[mat_m["severity"] == s])
        for d_ in range(D_CAP + 1):
            ec_rows.append({"severity": s, "label": SEV_LABELS[s], "d": d_,
                            "F": F[d_], "n": n})
    ec = pd.DataFrame(ec_rows)
    ec.to_csv(os.path.join(TAB_DIR, "ecdf_by_severity.csv"), index=False)

    for name, col in (("channel", mat_m["cmpl_type"].astype(str)), ("era", mat_m["era"])):
        rr = []
        for lvl, sub in mat_m.groupby(col, observed=True):
            if len(sub) < 100:
                continue
            F, n = ecdf_months(sub)
            for d_ in range(D_CAP + 1):
                rr.append({name: lvl, "d": d_, "F": F[d_], "n": n})
        pd.DataFrame(rr).to_csv(os.path.join(TAB_DIR, f"ecdf_by_{name}.csv"), index=False)

    # share reported within 1/3/6/12 months, empirical (mature)
    stats_out["empirical_F_by_severity"] = {
        SEV_LABELS[s]: {f"F({h})": float(ec.loc[(ec.severity == s) & (ec.d == h), "F"].iloc[0])
                        for h in (1, 3, 6, 12, 24, 36)}
        for s in sorted(mat_m["severity"].unique())
    }

    # ---------------------------------------------------- KS tests + bootstrap CIs
    ks, bt = {}, {}
    y0 = mat_d.loc[mat_d["severity"] == 0, "delay_days"].to_numpy()
    for s in (1, 2, 3):
        ys = mat_d.loc[mat_d["severity"] == s, "delay_days"].to_numpy()
        k = stats.ks_2samp(ys, y0, method="asymp")
        ks[f"Y{s}_vs_Y0"] = {"D": float(k.statistic), "p": float(k.pvalue),
                             "n1": int(len(ys)), "n2": int(len(y0))}
        bt[f"Y{s}_minus_Y0_median_days"] = boot_median_diff(ys, y0, rng)
    stats_out["ks_tests"] = ks
    stats_out["bootstrap_median_diffs"] = bt

    # ---------------------------------------------------- delay drift by cohort year
    drift = (mat_d.groupby(["cohort_year", "severity"], observed=True)["delay_days"]
             .agg(n="size", p25=lambda x: np.quantile(x, .25),
                  p50="median", p75=lambda x: np.quantile(x, .75)).reset_index())
    drift_all = (mat_d.groupby("cohort_year")["delay_days"]
                 .agg(n="size", p25=lambda x: np.quantile(x, .25),
                      p50="median", p75=lambda x: np.quantile(x, .75)).reset_index())
    drift_all["severity"] = -1
    pd.concat([drift, drift_all], ignore_index=True).to_csv(
        os.path.join(TAB_DIR, "delay_drift.csv"), index=False)
    stats_out["median_delay_by_era_days"] = {
        r["level"]: r["p50"] for r in rows if r["group"] == "era"}

    # ---------------------------------------------------- hazard model, eq (6)
    stats_out["hazard_model"] = hazard_model(
        mat_m, os.path.join(TAB_DIR, "hazard_model.csv"))

    # ---------------------------------------------------- Gate 2 reconciliation grid
    # The outline quotes "fatality median ~132 days vs ~20 overall" (verified 2026-07-23).
    # That value comes from an ALL-CHANNELS set with NO 1995 cohort floor. Our main
    # spec is consumer-only with the floor (pre-1995 cohorts are left-truncated: their
    # complaints received before 1995-01 are absent from the snapshot, which biases
    # their observed delays LONG). Both variants are computed here so the difference is
    # documented rather than assumed.
    recon = []
    for cons in (True, False):
        for floor in (True, False):
            m = df["valid_delay"] & (df["cohort_p"] <= MATURE_MAX) & (df["delay_days"] <= DAY_CAP)
            if cons:
                m &= df["consumer"]
            if floor:
                m &= df["cohort_p"] >= 1995 * 12
            sub = df.loc[m]
            recon.append({
                "channels": "consumer" if cons else "all",
                "cohort_floor_1995": floor,
                "n": int(len(sub)),
                "median_all_days": float(np.median(sub["delay_days"])),
                "median_Y0_days": float(np.median(sub.loc[sub.severity == 0, "delay_days"])),
                "median_Y3_days": float(np.median(sub.loc[sub.severity == 3, "delay_days"])),
                "n_Y3": int((sub.severity == 3).sum()),
                "is_main_spec": bool(cons and floor),
            })
    stats_out["gate2_reconciliation"] = {
        "note": ("Outline's ~132 d fatality median reproduces under channels=all + "
                 "no 1995 cohort floor. Main spec (consumer + floor) gives a smaller "
                 "but still ~5x gradient; the floor removes left-truncated pre-1995 "
                 "cohorts whose delays are biased long."),
        "grid": recon,
    }

    # ---------------------------------------------------- Gate 2
    med_all = float(np.median(mat_d["delay_days"]))
    med = {s: float(np.median(mat_d.loc[mat_d["severity"] == s, "delay_days"]))
           for s in sorted(mat_d["severity"].unique())}
    stats_out["gate2"] = {"median_overall_days": med_all,
                          "median_by_severity_days": {f"Y{k}": v for k, v in med.items()},
                          "ratio_Y3_over_Y0": med[3] / med[0] if med.get(0) else None}
    g_a = 10 <= med_all <= 30
    g_b = med[3] >= 4 * med[0]
    qa_log("2", "Gate 2  -  overall median delay 2-3 weeks", g_a, f"{med_all:.0f} days")
    qa_log("2", "Gate 2  -  median(Y3) >= 4 x median(Y0)", g_b,
           f"Y3 {med[3]:.0f} d vs Y0 {med[0]:.0f} d (ratio {med[3] / med[0]:.2f})")
    _allnf = [r for r in recon if r["channels"] == "all" and not r["cohort_floor_1995"]][0]
    qa_log("2", "Gate 2  -  outline's ~132 d Y3 median reproduces in its own variant",
           125 <= _allnf["median_Y3_days"] <= 140,
           f"all-channels / no-1995-floor: Y3 {_allnf['median_Y3_days']:.0f} d, "
           f"overall {_allnf['median_all_days']:.0f} d; main spec (consumer+floor) "
           f"Y3 {med[3]:.0f} d  -  difference documented in deviations.md")

    write_json(os.path.join(AUDIT_DIR, "delay_stats.json"), stats_out)

    print(f"  mature consumer complaints: day-level {len(mat_d):,}  month-level {len(mat_m):,}")
    print(f"  median delay overall {med_all:.0f} d; " +
          "; ".join(f"Y{k} {v:.0f} d" for k, v in med.items()))
    hr = stats_out["hazard_model"]["severity_hazard_ratios"]
    for k, v in hr.items():
        print(f"  hazard ratio {k}: {v['hr']:.3f} [{v['lo95']:.3f}, {v['hi95']:.3f}]")
    if not (g_a and g_b):
        raise SystemExit("GATE 2 FAILED  -  severity/delay gradient absent; stop.")
    print("  GATE 2 PASSED")


if __name__ == "__main__":
    main()
