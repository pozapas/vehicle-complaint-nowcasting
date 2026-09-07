"""Step 1  -  complaint-level build.

Streams the seven NHTSA flat files, computes the ROW-LEVEL audit aggregates
(Gate 1 targets are row-level), then collapses to complaint level by ODINO:
earliest LDATE, first nonmissing FAILDATE, max severity across rows.

Outputs
  Data/processed/complaints_nowcast.parquet
  results/audits/build_audit.json
"""

from __future__ import annotations

import glob
import os
import sys
import time
from operator import itemgetter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (AUDIT_DIR, CONSUMER_CHANNELS, D_CAP, PARQUET, RAW_DIR,
                    T_CUTOFF, month_index, qa_log, write_json)

# field positions (0-indexed) per outline Step 1 / SCHEMA.md
COLS = {
    "odino": 1, "maketxt": 3, "yeartxt": 5, "crash": 6, "faildate": 7,
    "fire": 8, "injured": 9, "deaths": 10, "compdesc": 11, "ldate": 16,
    "cmpl_type": 20,
}
NAMES = list(COLS)
GET = itemgetter(*COLS.values())
NFIELDS = 51


def parse_file(path, audit):
    """Parse one flat file defensively -> typed row-level DataFrame."""
    rows = []
    n_lines = 0
    n_bad = 0
    with open(path, "r", encoding="latin-1", errors="replace", newline="") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            n_lines += 1
            parts = line.split("\t")
            if len(parts) != NFIELDS:
                n_bad += 1
                continue
            rows.append(GET(parts))

    audit["rows_read"] += n_lines
    audit["rows_bad_fieldcount"] += n_bad
    audit["per_file"].append({
        "file": os.path.basename(path), "lines": n_lines, "bad_fieldcount": n_bad,
        "kept": len(rows),
    })

    cols = list(zip(*rows)) if rows else [()] * len(NAMES)
    del rows
    df = pd.DataFrame({n: pd.array(c, dtype="string") for n, c in zip(NAMES, cols)})
    del cols

    out = pd.DataFrame(index=df.index)
    out["odino"] = pd.to_numeric(df["odino"], errors="coerce").astype("Int64")
    out["ldate_i"] = pd.to_numeric(df["ldate"], errors="coerce").astype("Int64")
    out["faildate_i"] = pd.to_numeric(df["faildate"], errors="coerce").astype("Int64")
    out["ldate_dt"] = pd.to_datetime(df["ldate"], format="%Y%m%d", errors="coerce")
    out["faildate_dt"] = pd.to_datetime(df["faildate"], format="%Y%m%d", errors="coerce")

    deaths = pd.to_numeric(df["deaths"], errors="coerce").fillna(0)
    injured = pd.to_numeric(df["injured"], errors="coerce").fillna(0)
    crash = df["crash"].fillna("").str.upper().str.strip().eq("Y")
    fire = df["fire"].fillna("").str.upper().str.strip().eq("Y")

    sev = np.zeros(len(df), dtype="int8")
    sev[(crash | fire).to_numpy()] = 1
    sev[(injured > 0).to_numpy()] = 2
    sev[(deaths > 0).to_numpy()] = 3
    out["severity"] = sev
    out["crash"] = crash.to_numpy()
    out["fire"] = fire.to_numpy()
    out["injured"] = injured.astype("int32").to_numpy()
    out["deaths"] = deaths.astype("int32").to_numpy()

    out["make"] = df["maketxt"].fillna("").str.upper().str.strip()
    out["yeartxt"] = df["yeartxt"].fillna("").str.strip()
    out["cmpl_type"] = df["cmpl_type"].fillna("").str.upper().str.strip()
    # component family = text before the first ':' in COMPDESC
    comp = df["compdesc"].fillna("").str.upper().str.strip()
    out["comp_family"] = comp.str.split(":", n=1).str[0].str.strip()
    del df
    return out


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "COMPLAINTS_RECEIVED_*.txt")))
    if len(files) != 7:
        raise SystemExit(f"expected 7 flat files, found {len(files)} in {RAW_DIR}")

    audit = {"snapshot": "nhtsa_20260703", "files": [os.path.basename(f) for f in files],
             "rows_read": 0, "rows_bad_fieldcount": 0, "per_file": []}

    frames = []
    for f in files:
        print(f"  parsing {os.path.basename(f)} ...", flush=True)
        frames.append(parse_file(f, audit))
    rows = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  parsed {len(rows):,} rows in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------ row-level audit
    both = rows["ldate_dt"].notna() & rows["faildate_dt"].notna()
    delay_days_row = (rows["ldate_dt"] - rows["faildate_dt"]).dt.days
    neg_row = (delay_days_row < 0).fillna(False)

    sev_row = rows["severity"].value_counts().sort_index()
    audit["row_level"] = {
        "rows_kept": int(len(rows)),
        "unique_odino": int(rows["odino"].nunique(dropna=True)),
        "ldate_missing": int(rows["ldate_dt"].isna().sum()),
        "faildate_missing": int(rows["faildate_dt"].isna().sum()),
        "both_dates": int(both.sum()),
        "both_dates_share": float(both.mean()),
        "negative_delay": int(neg_row.sum()),
        "severity_counts": {f"Y{k}": int(v) for k, v in sev_row.items()},
        "crash_rows": int(rows["crash"].sum()),
        "fire_rows": int(rows["fire"].sum()),
        "ldate_min": int(rows["ldate_i"].min()), "ldate_max": int(rows["ldate_i"].max()),
        "cmpl_type_counts": {str(k): int(v) for k, v in
                             rows["cmpl_type"].value_counts().head(15).items()},
    }

    # ------------------------------------------------------------ Gate 1
    rl = audit["row_level"]
    g1 = []
    g1.append(("rows == 2,221,663", audit["rows_read"] == 2_221_663, f"{audit['rows_read']:,}"))
    g1.append(("unique ODINO == 1,603,201", rl["unique_odino"] == 1_603_201,
               f"{rl['unique_odino']:,}"))
    g1.append(("both-dates share ~95%", 0.93 <= rl["both_dates_share"] <= 0.97,
               f"{rl['both_dates_share']:.4f}"))
    g1.append(("negative delays ~582", 500 <= rl["negative_delay"] <= 700,
               f"{rl['negative_delay']:,}"))
    g1.append(("Y1 ~117k", 100_000 <= rl["severity_counts"].get("Y1", 0) <= 140_000,
               f"{rl['severity_counts'].get('Y1', 0):,}"))
    g1.append(("Y2 ~84k", 70_000 <= rl["severity_counts"].get("Y2", 0) <= 100_000,
               f"{rl['severity_counts'].get('Y2', 0):,}"))
    g1.append(("Y3 ~4.1k", 3_000 <= rl["severity_counts"].get("Y3", 0) <= 5_500,
               f"{rl['severity_counts'].get('Y3', 0):,}"))
    audit["gate1"] = [{"check": c, "pass": bool(p), "value": v} for c, p, v in g1]
    for c, p, v in g1:
        qa_log("1", f"Gate 1  -  {c}", p, f"observed {v}")

    # ------------------------------------------------------------ collapse by ODINO
    rows = rows[rows["odino"].notna()].copy()
    rows["odino"] = rows["odino"].astype("int64")
    print("  collapsing to complaint level ...", flush=True)

    g = rows.groupby("odino", sort=True)
    c = pd.DataFrame({
        "ldate_dt": g["ldate_dt"].min(),          # earliest LDATE
        "faildate_dt": g["faildate_dt"].first(),  # first NONMISSING FAILDATE
        "severity": g["severity"].max(),          # max severity across rows
        "n_rows": g.size().astype("int16"),
        "make": g["make"].first(),
        "yeartxt": g["yeartxt"].first(),
        "cmpl_type": g["cmpl_type"].first(),
        "comp_family": g["comp_family"].first(),
        "deaths": g["deaths"].max(),
        "injured": g["injured"].max(),
        "crash": g["crash"].max(),
        "fire": g["fire"].max(),
    }).reset_index()
    del rows, g

    # ------------------------------------------------------------ derived fields
    c["ldate_i"] = (c["ldate_dt"].dt.year * 10000 + c["ldate_dt"].dt.month * 100
                    + c["ldate_dt"].dt.day).astype("Int64")
    c["faildate_i"] = (c["faildate_dt"].dt.year * 10000 + c["faildate_dt"].dt.month * 100
                       + c["faildate_dt"].dt.day).astype("Int64")

    c["report_p"] = month_index(c["ldate_i"].astype("float64"))
    c["cohort_p"] = month_index(c["faildate_i"].astype("float64"))
    # eq (1): d = 12*(yL - yF) + (mL - mF)  -- integer month arithmetic
    c["delay_m"] = c["report_p"] - c["cohort_p"]
    c["delay_days"] = (c["ldate_dt"] - c["faildate_dt"]).dt.days

    c["has_both_dates"] = c["ldate_dt"].notna() & c["faildate_dt"].notna()
    c["neg_delay"] = (c["delay_days"] < 0).fillna(False)
    c["consumer"] = c["cmpl_type"].isin(CONSUMER_CHANNELS)
    c["pre2003"] = (c["cohort_p"] < 2003 * 12).fillna(False)
    c["beyond_cap"] = (c["delay_m"] > D_CAP).fillna(False)
    c["valid_delay"] = c["has_both_dates"] & (~c["neg_delay"])
    c["mature"] = c["valid_delay"] & (c["cohort_p"] <= T_CUTOFF - D_CAP)
    # analysis flag: in-triangle observation (cohort in range, delay within cap)
    c["in_triangle"] = (c["valid_delay"] & (c["delay_m"] <= D_CAP)
                        & (c["report_p"] <= T_CUTOFF) & (c["cohort_p"] >= 1995 * 12))

    for col in ("report_p", "cohort_p", "delay_m"):
        c[col] = c[col].astype("float32")
    c["delay_days"] = c["delay_days"].astype("float32")
    for col in ("make", "cmpl_type", "comp_family", "yeartxt"):
        c[col] = c[col].astype("category")

    c = c.drop(columns=["ldate_dt", "faildate_dt"])
    c.to_parquet(PARQUET, index=False, compression="zstd")

    # ------------------------------------------------------------ complaint-level audit
    cons = c[c["consumer"]]
    mature_cons = cons[cons["mature"]]
    beyond = mature_cons["beyond_cap"]
    audit["complaint_level"] = {
        "complaints": int(len(c)),
        "multi_row_complaints": int((c["n_rows"] > 1).sum()),
        "max_rows_per_odino": int(c["n_rows"].max()),
        "both_dates": int(c["has_both_dates"].sum()),
        "both_dates_share": float(c["has_both_dates"].mean()),
        "negative_delay": int(c["neg_delay"].sum()),
        "severity_counts": {f"Y{k}": int(v) for k, v in
                            c["severity"].value_counts().sort_index().items()},
        "consumer_complaints": int(c["consumer"].sum()),
        "consumer_share": float(c["consumer"].mean()),
        "channel_counts": {str(k): int(v) for k, v in
                           c["cmpl_type"].value_counts().head(15).items()},
        "pre2003_cohort": int(c["pre2003"].sum()),
        "in_triangle_consumer": int(cons["in_triangle"].sum()),
        "mature_consumer": int(len(mature_cons)),
        "beyond_cap_share_mature_consumer": float(beyond.mean()) if len(mature_cons) else None,
        "beyond_cap_share_mature_consumer_by_severity": {
            f"Y{s}": float(mature_cons.loc[mature_cons["severity"] == s, "beyond_cap"].mean())
            for s in sorted(mature_cons["severity"].unique())
        },
        "T_cutoff": "2026-06",
        "dropped_report_month_2026_07": int((c["report_p"] > T_CUTOFF).sum()),
    }
    audit["elapsed_sec"] = round(time.time() - t0, 1)
    write_json(os.path.join(AUDIT_DIR, "build_audit.json"), audit)

    print(f"\n  complaints: {len(c):,}  consumer: {c['consumer'].sum():,}")
    print(f"  beyond-cap share (mature consumer): "
          f"{audit['complaint_level']['beyond_cap_share_mature_consumer']:.4f}")
    print(f"  done in {audit['elapsed_sec']}s -> {PARQUET}")

    failed = [x for x in audit["gate1"] if not x["pass"]]
    if failed:
        raise SystemExit(f"GATE 1 FAILED: {failed}")
    print("  GATE 1 PASSED")


if __name__ == "__main__":
    main()
