"""Step 8  -  robustness battery (outline §4.6).

Variants: all channels; D = 24; W = 36 and all-history; exclude pre-2003 cohorts;
exclude recall/publicity burst months; delay measured by elapsed days rather than the
calendar-month difference of eq (1).

For each variant we report the undercount ratios F_hat_s(h) (the paper's key
descriptive) and a reduced rolling-origin backtest (relative bias + pooled 90%
coverage), always against the same base specification.

Outputs
  results/audits/robustness.csv
  results/audits/robustness_audit.json
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chainladder as cl
from common import (AUDIT_DIR, BT_FIRST, BT_LAST, COVERAGE_FIRST, D_CAP, PARQUET,
                    PHI_FIT_LAST, SEED, T_CUTOFF, W_DEFAULT, p_to_str, write_json)

_spec = importlib.util.spec_from_file_location(
    "bt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

T0 = cl.T0_FLOOR
STREAMS = ("Y0", "Y3", "severe")
HS = (1, 3, 6, 12)
B_ROB = 200          # reduced from 500 for the battery; logged in the audit


def stream_mask(df, name):
    sev = df["severity"].to_numpy()
    if name == "severe":
        return None, sev >= 1
    return int(name[1]), None


def make_variant(df, key):
    """Return (frame, D, W) for a named variant. `df` is the full complaint table."""
    d = df
    D, W = D_CAP, W_DEFAULT
    if key == "base":
        d = df[df["consumer"]]
    elif key == "all_channels":
        d = df
    elif key == "D24":
        d, D = df[df["consumer"]], 24
    elif key == "W36":
        d, W = df[df["consumer"]], 36
    elif key == "W_all_history":
        d, W = df[df["consumer"]], 10_000
    elif key == "exclude_pre2003":
        d = df[df["consumer"] & (df["cohort_p"] >= 2003 * 12)]
    elif key == "exclude_burst_months":
        d = df[df["consumer"] & ~burst_flag(df[df["consumer"]])
               .reindex(df.index, fill_value=False)]
    elif key == "delay_by_elapsed_days":
        d = df[df["consumer"]].copy()
        # eq (1) uses the calendar-month difference; here delay is elapsed days / 30.44
        d["delay_m"] = np.floor(d["delay_days"] / 30.44).astype("float32")
        d["report_p"] = d["cohort_p"] + d["delay_m"]
    else:
        raise KeyError(key)
    return d, D, W


def burst_flag(cons):
    """Flag complaints in make x component-family x report-month spikes.

    A cell is a burst if it holds >= 20 complaints AND more than 10x that stream's
    median nonzero monthly volume  -  the signature of a recall/attorney batch filing
    rather than organic consumer reporting.
    """
    g = cons.groupby([cons["make"].astype(str), cons["comp_family"].astype(str),
                      cons["report_p"]], observed=True).size().rename("n").reset_index()
    med = g.groupby(["make", "comp_family"], observed=True)["n"].median().rename("med")
    g = g.merge(med, on=["make", "comp_family"], how="left")
    g["burst"] = (g["n"] >= 20) & (g["n"] > 10 * g["med"].clip(lower=1))
    keys = set(map(tuple, g.loc[g["burst"], ["make", "comp_family", "report_p"]].values))
    if not keys:
        return pd.Series(False, index=cons.index)
    tup = list(zip(cons["make"].astype(str), cons["comp_family"].astype(str),
                   cons["report_p"]))
    return pd.Series([t in keys for t in tup], index=cons.index)


def run_variant(df, key, rng):
    d, D, W = make_variant(df, key)
    tris = {}
    for s in STREAMS:
        sev, msk = stream_mask(d, s)
        tris[s] = cl.build_triangle(d, T_CUTOFF, D, t0=T0, severity=sev, mask=msk)

    out = {"variant": key, "D": D, "W": W, "n_complaints": int(len(d)), "F": {},
           "bias": {}, "cov90": {}}
    for s in STREAMS:
        _, F = cl.fit_F(tris[s], T_CUTOFF, D, W, t0=T0)
        out["F"][s] = {h: float(F[min(h, D)]) for h in HS}

    # ---- reduced backtest, same censoring gate and phi fit as Step 4
    recs, draws = [], []
    realized = {s: np.cumsum(tris[s], axis=1)[:, D] for s in STREAMS}
    for Ts in range(BT_FIRST, BT_LAST + 1):
        for s in STREAMS:
            tri_c = cl.censor(tris[s], Ts, D, t0=T0)
            _, F = cl.fit_F(tri_c, Ts, D, W, t0=T0)
            Fb = cl.bootstrap_F_multinomial(tri_c, Ts, D, W, B_ROB, rng, t0=T0)
            t, C_obs, N_hat, h = cl.nowcast(tri_c, F, Ts, D, t0=T0)
            for i in np.where(t >= Ts - 24)[0]:
                ti = int(t[i])
                recs.append({"Tstar": Ts, "stream": s, "h": int(h[i]),
                             "C_obs": float(C_obs[i]), "N_hat": float(N_hat[i]),
                             "realized": float(realized[s][ti - T0]),
                             "train": bool(Ts <= PHI_FIT_LAST)})
                draws.append((C_obs[i] / np.maximum(Fb[:, h[i]], 1e-12)))
    obs = pd.DataFrame(recs)
    phi, _ = bt.fit_uncertainty(obs, draws)
    obs = bt.simulate(obs, draws, phi, rng)
    ev = obs[(obs["Tstar"] >= COVERAGE_FIRST) & (obs["realized"] > 0)]
    for s in STREAMS:
        g = ev[ev.stream == s]
        out["cov90"][s] = float(g[g["h"].isin(HS)]["cov90"].mean())
        out["bias"][s] = {}
        for h in HS:
            gg = g[g["h"] == h]
            if len(gg):
                rel = (gg["N_hat"] - gg["realized"]) / gg["realized"]
                out["bias"][s][h] = float(rel.mean())
    return out


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PARQUET)
    df = df[df["valid_delay"]].copy()

    variants = ["base", "all_channels", "D24", "W36", "W_all_history",
                "exclude_pre2003", "exclude_burst_months", "delay_by_elapsed_days"]
    res = {}
    for k in variants:
        print(f"  variant {k} ...", flush=True)
        res[k] = run_variant(df, k, rng)

    base = res["base"]
    rows = []
    for k, r in res.items():
        for s in STREAMS:
            for h in HS:
                rows.append({"variant": k, "quantity": "F_hat(h)", "stream": s, "h": h,
                             "value": r["F"][s][h],
                             "delta_vs_base": r["F"][s][h] - base["F"][s][h]})
                if h in r["bias"].get(s, {}):
                    rows.append({"variant": k, "quantity": "nowcast_rel_bias",
                                 "stream": s, "h": h, "value": r["bias"][s][h],
                                 "delta_vs_base": r["bias"][s][h]
                                 - base["bias"][s].get(h, np.nan)})
            rows.append({"variant": k, "quantity": "pooled_cov90", "stream": s,
                         "h": np.nan, "value": r["cov90"][s],
                         "delta_vs_base": r["cov90"][s] - base["cov90"][s]})
        rows.append({"variant": k, "quantity": "n_complaints", "stream": "-",
                     "h": np.nan, "value": r["n_complaints"],
                     "delta_vs_base": r["n_complaints"] - base["n_complaints"]})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(AUDIT_DIR, "robustness.csv"), index=False)

    fh = rdf[(rdf.quantity == "F_hat(h)")]
    audit = {"B_bootstrap": B_ROB,
             "note": ("B reduced from 500 to 200 for the battery only; the headline "
                      "backtest in Step 4 uses B = 500."),
             "variants": {k: {"D": v["D"], "W": v["W"], "n_complaints": v["n_complaints"],
                              "F": v["F"], "cov90": v["cov90"], "bias": v["bias"]}
                          for k, v in res.items()},
             "max_abs_delta_F_vs_base": {
                 k: float(fh[fh.variant == k]["delta_vs_base"].abs().max())
                 for k in variants},
             "max_abs_delta_F_Y3_h3": {
                 k: float(fh[(fh.variant == k) & (fh.stream == "Y3")
                             & (fh.h == 3)]["delta_vs_base"].iloc[0])
                 for k in variants}}
    write_json(os.path.join(AUDIT_DIR, "robustness_audit.json"), audit)

    print("\n  F_hat_Y3(3) by variant (base = "
          f"{base['F']['Y3'][3]:.3f}):")
    for k in variants:
        print(f"    {k:<24} {res[k]['F']['Y3'][3]:.3f}  "
              f"(delta {res[k]['F']['Y3'][3] - base['F']['Y3'][3]:+.3f})")
    print("\n  max |delta F_hat| vs base across all streams/horizons:")
    for k in variants:
        print(f"    {k:<24} {audit['max_abs_delta_F_vs_base'][k]:.3f}")


if __name__ == "__main__":
    main()
