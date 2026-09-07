"""Shared paths, constants and month-index helpers for the TRB nowcasting analysis.

Every script imports from here so that `T`, the delay cap `D`, the maturity rule
and the consumer-channel definition exist in exactly one place.
"""

from __future__ import annotations

import json
import os

import numpy as np

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "Data", "external", "raw", "nhtsa_20260703", "extracted")
PROC_DIR = os.path.join(ROOT, "Data", "processed")
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
GRAY_DIR = os.path.join(FIG_DIR, "gray")
TAB_DIR = os.path.join(RESULTS, "tables")
AUDIT_DIR = os.path.join(RESULTS, "audits")
TRB_TEMPLATE_DIR = os.path.join(
    ROOT, "Unofficial Transportation Research Board (TRB) LaTeX template"
)

PARQUET = os.path.join(PROC_DIR, "complaints_nowcast.parquet")

for _d in (PROC_DIR, FIG_DIR, GRAY_DIR, TAB_DIR, AUDIT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- analysis constants
# Month index p = year*12 + (month-1); see month_index().
D_CAP = 36                       # delay cap in months (outline sec 4.1)
T_CUTOFF = 2026 * 12 + 5         # 2026-06; July 2026 dropped (partial month)
MATURE_MAX = T_CUTOFF - D_CAP    # 2023-06: last fully mature incident cohort

CONSUMER_CHANNELS = ("VOQ", "EVOQ", "IVOQ", "MAVQ")

SEV_LABELS = {0: "Y0 none", 1: "Y1 crash/fire", 2: "Y2 injury", 3: "Y3 fatality"}
SEV_SHORT = {0: "Y0", 1: "Y1", 2: "Y2", 3: "Y3"}
SEV_ORDER = (0, 1, 2, 3)

ERAS = (("1995-2004", 1995, 2004), ("2005-2014", 2005, 2014), ("2015-2023", 2015, 2023))

HORIZONS = (1, 3, 6, 12, 24)

# Backtest origins (outline sec 4.4): monthly pseudo-cutoffs.
BT_FIRST = 2018 * 12 + 0   # 2018-01
BT_LAST = 2023 * 12 + 5    # 2023-06
PHI_FIT_LAST = 2020 * 12 + 11   # phi-hat fitted on origins <= 2020-12
COVERAGE_FIRST = 2021 * 12 + 0  # coverage evaluated on origins >= 2021-01

W_DEFAULT = 60   # rolling window (months) for development factors
N_BOOT = 500     # cohort bootstrap replicates
SEED = 20260723


# ---------------------------------------------------------------- month helpers
def month_index(yyyymmdd):
    """YYYYMMDD integer(s) -> month index p = year*12 + (month-1). NaN-safe."""
    a = np.asarray(yyyymmdd, dtype="float64")
    y = np.floor(a / 10000.0)
    m = np.floor(a / 100.0) % 100.0
    return y * 12.0 + (m - 1.0)


def p_to_year(p):
    return np.asarray(p, dtype="int64") // 12


def p_to_month(p):
    return np.asarray(p, dtype="int64") % 12 + 1


def p_to_str(p):
    p = int(p)
    return f"{p // 12:04d}-{p % 12 + 1:02d}"


def str_to_p(s):
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


# ---------------------------------------------------------------- io helpers
def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_jsonify)
    return path


def _jsonify(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def qa_log(step, gate, passed, detail):
    """Append one gate result to results/audits/qa.md."""
    path = os.path.join(AUDIT_DIR, "qa.md")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if new:
            fh.write("# QA log  -  gate results and fallbacks\n\n")
            fh.write("Written by code (`common.qa_log`). One row per gate check.\n\n")
            fh.write("| Step | Gate | Result | Detail |\n|---|---|---|---|\n")
        mark = "PASS" if passed else "**FAIL**"
        fh.write(f"| {step} | {gate} | {mark} | {detail} |\n")
    return passed
