"""Chain-ladder / reporting-triangle machinery  -  outline eqs (1)-(5), (7), (8).

Every downstream script (nowcast, backtest, case studies, robustness) goes through
these functions. In particular `censor()` is the ONLY way a pseudo-cutoff is applied,
so there is exactly one place where backtest leakage could occur.

Conventions
  t  : incident-cohort month index p = year*12 + (month-1)
  d  : reporting delay in whole months, 0..D
  T  : data-cutoff month; cell (t, d) is observed iff t + d <= T
  Triangles are numpy arrays n[i, d] where i = t - t0 (t0 = first cohort).
"""

from __future__ import annotations

import numpy as np

T0_FLOOR = 1995 * 12   # snapshot starts 1995-01; earlier cohorts are left-truncated


# ------------------------------------------------------------------ construction
def build_triangle(df, T, D, t0=T0_FLOOR, severity=None, mask=None):
    """Complaint-level frame -> counts n[i, d]. Uses only in-cap, valid, t+d <= T rows."""
    m = (df["valid_delay"].to_numpy()
         & (df["cohort_p"].to_numpy() >= t0)
         & (df["cohort_p"].to_numpy() <= T)
         & (df["delay_m"].to_numpy() >= 0)
         & (df["delay_m"].to_numpy() <= D)
         & (df["report_p"].to_numpy() <= T))
    if severity is not None:
        m &= (df["severity"].to_numpy() == severity)
    if mask is not None:
        m &= np.asarray(mask)
    t = df["cohort_p"].to_numpy()[m].astype(np.int64) - t0
    d = df["delay_m"].to_numpy()[m].astype(np.int64)
    n_t = int(T - t0) + 1
    tri = np.zeros((n_t, D + 1), dtype=np.float64)
    np.add.at(tri, (t, d), 1.0)
    return tri


def censor(tri, Tstar, D, t0=T0_FLOOR):
    """Apply a pseudo-cutoff: zero every cell with t + d > Tstar. THE leakage gate.

    Returns a (n_t_star, D+1) triangle covering cohorts t0..Tstar only. Nothing
    downstream of this call can see a complaint that had not arrived by Tstar.
    """
    n_t = int(Tstar - t0) + 1
    out = tri[:n_t, : D + 1].copy()
    i = np.arange(n_t)[:, None]
    d = np.arange(D + 1)[None, :]
    out[(i + t0 + d) > Tstar] = 0.0
    return out


# ------------------------------------------------------------------ eqs (4)-(5)
def dev_factors(tri, T, D, W, t0=T0_FLOOR, log=None):
    """Eq (4): lambda_hat(d) = sum_{t in W_d} C(t,d+1) / sum_{t in W_d} C(t,d).

    W_d = {t : t + d + 1 <= T, t >= T - d - 1 - W}   (both cells observed, <= W back).
    Zero denominators trigger a per-d window extension (outline pitfall 5), logged.
    """
    C = np.cumsum(tri, axis=1)
    lam = np.empty(D, dtype=np.float64)
    for d in range(D):
        tmax = T - d - 1
        w = W
        while True:
            tmin = max(t0, T - d - 1 - w)
            if tmax < tmin:
                lam[d] = np.nan
                break
            lo, hi = int(tmin - t0), int(tmax - t0) + 1
            den = C[lo:hi, d].sum()
            if den > 0:
                lam[d] = C[lo:hi, d + 1].sum() / den
                if w != W and log is not None:
                    log.append({"d": int(d), "window_extended_to": int(w),
                                "reason": "zero denominator at requested W"})
                break
            if tmin <= t0:
                lam[d] = np.nan
                if log is not None:
                    log.append({"d": int(d), "window_extended_to": "all-history",
                                "reason": "zero denominator even with all history"})
                break
            w = w * 2 if w > 0 else 12
    # a delay step with no information at all: assume no further development
    lam = np.where(np.isfinite(lam) & (lam >= 1.0), lam, 1.0)
    return lam


def F_from_lambda(lam, D):
    """Eq (5): F_hat(d) = prod_{u=d}^{D-1} lambda_hat(u)^-1, F_hat(D) = 1."""
    F = np.ones(D + 1, dtype=np.float64)
    for d in range(D - 1, -1, -1):
        F[d] = F[d + 1] / lam[d]
    return F


def fit_F(tri, T, D, W, t0=T0_FLOOR, log=None):
    """Convenience: triangle -> (lambda_hat, F_hat)."""
    lam = dev_factors(tri, T, D, W, t0=t0, log=log)
    return lam, F_from_lambda(lam, D)


# ------------------------------------------------------------------ eqs (7)-(8)
def nowcast(tri, F, T, D, t0=T0_FLOOR):
    """Eq (7): N_hat(t) = C(t, h) / F_hat(h) with h = min(T - t, D).

    For mature cohorts (T - t >= D) this returns exactly C(t, D) because F(D) = 1.
    Returns (t_index, C_obs, N_hat, h).
    """
    C = np.cumsum(tri, axis=1)
    n_t = tri.shape[0]
    t = np.arange(n_t) + t0
    h = np.minimum(T - t, D).astype(int)
    C_obs = C[np.arange(n_t), h]
    N_hat = C_obs / F[h]
    return t, C_obs, N_hat, h


def undercount_curve(F, D):
    """Eq (8): R(h) = F_hat(h), the visible fraction h months after the incident month."""
    return np.arange(D + 1), F.copy()


# ------------------------------------------------- one-step empirical column ratio
def F_onestep(tri, T, D, W, t0=T0_FLOOR):
    """Chain-ladder-free maturity curve read straight off fully mature cohorts.

    F_1(d) = sum_{t in M} C(t,d) / sum_{t in M} C(t,D),
        M = {t : t + D <= T, t >= T - D - W}.

    This is the obvious alternative to eqs (4)-(5): instead of chaining D adjacent
    column ratios, it takes a single ratio of the cumulative-at-d to the ultimate,
    pooled over cohorts that are already complete. It makes no assumption that
    development is homogeneous across cohorts, but it can only see cohorts at least
    D months old, so under a drifting delay distribution it is estimating a curve
    that is D to D+W months stale. Comparing the two settles whether the chaining
    step earns its place.

    Uses only cells with t + D <= T, so it is strictly inside the censored triangle
    and cannot leak.
    """
    C = np.cumsum(tri, axis=1)
    tmax, tmin = T - D, max(t0, T - D - W)
    lo, hi = int(tmin - t0), int(tmax - t0) + 1
    if hi <= lo or hi <= 0:
        return np.ones(D + 1, dtype=np.float64)
    lo = max(lo, 0)
    den = C[lo:hi, D].sum()
    if den <= 0:
        return np.ones(D + 1, dtype=np.float64)
    F = C[lo:hi, :].sum(axis=0) / den
    F = np.minimum(np.maximum.accumulate(F), 1.0)   # monotone, capped at 1
    F[D] = 1.0
    return F


# ------------------------------------------------------------------ bootstrap
def bootstrap_F(tri, T, D, W, B, rng, t0=T0_FLOOR):
    """Nonparametric bootstrap over COHORTS in the estimation window (outline 4.4-i).

    Each replicate draws multinomial weights over cohorts ONCE and reuses them for
    every delay step, so dependence across d is preserved. Weighting (rather than
    relocating rows) keeps each cohort paired with its own censoring pattern --
    a shuffled row placed at a different position would carry the wrong triangle
    boundary and silently corrupt lambda_hat.

    Returns (B, D+1) array of F_hat draws.
    """
    C = np.cumsum(tri, axis=1)
    n_t = tri.shape[0]
    lo_all = max(0, int(T - t0) - (W + D + 1))
    k = n_t - lo_all
    if k <= 1:
        return np.tile(F_from_lambda(dev_factors(tri, T, D, W, t0=t0), D), (B, 1))
    Wt = rng.multinomial(k, np.full(k, 1.0 / k), size=B).astype(np.float64)

    lam_b = np.ones((B, D), dtype=np.float64)
    for d in range(D):
        tmax, tmin = T - d - 1, max(t0, T - d - 1 - W)
        lo, hi = max(int(tmin - t0), lo_all), int(tmax - t0) + 1
        if hi <= lo:
            continue
        w = Wt[:, lo - lo_all: hi - lo_all]
        den = w @ C[lo:hi, d]
        num = w @ C[lo:hi, d + 1]
        lam_b[:, d] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    lam_b = np.maximum(lam_b, 1.0)

    F = np.ones((B, D + 1), dtype=np.float64)
    for d in range(D - 1, -1, -1):
        F[:, d] = F[:, d + 1] / lam_b[:, d]
    return F


def _lambda_from_Cb(Cb, T, D, W, t0, lo_all):
    """Vectorized eq (4) over a stack of bootstrap cumulative triangles Cb (B, npool, D+1)."""
    B = Cb.shape[0]
    lam = np.ones((B, D), dtype=np.float64)
    for d in range(D):
        tmax, tmin = T - d - 1, max(t0, T - d - 1 - W)
        lo, hi = max(int(tmin - t0), lo_all), int(tmax - t0) + 1
        if hi <= lo:
            continue
        sl = slice(lo - lo_all, hi - lo_all)
        den = Cb[:, sl, d].sum(axis=1)
        num = Cb[:, sl, d + 1].sum(axis=1)
        lam[:, d] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    return np.maximum(lam, 1.0)


def bootstrap_F_multinomial(tri, T, D, W, B, rng, t0=T0_FLOOR):
    """Parametric bootstrap of F_hat under the model that defines the estimator.

    Section 4.2 states the delay model IS multinomial: given a cohort's eventual
    (delay-capped) total, its delays are multinomial with probabilities p(d). This
    bootstrap resamples each cohort's OBSERVED cells from the corresponding
    truncated multinomial -- multinomial(C(t,h), p(0..h)/F(h)) -- and refits eqs
    (4)-(5). It therefore isolates sampling error in F_hat, which is what component
    (i) of the interval is meant to represent; between-cohort drift is left to be
    absorbed by the fitted dispersion phi_hat in component (ii), so the two
    components stay disjoint instead of double-counting.

    Returns (B, D+1) array of F_hat draws.
    """
    C = np.cumsum(tri, axis=1)
    n_t = tri.shape[0]
    lo_all = max(0, int(T - t0) - (W + D + 1))
    npool = n_t - lo_all
    if npool <= 1:
        return np.tile(F_from_lambda(dev_factors(tri, T, D, W, t0=t0), D), (B, 1))

    F0 = F_from_lambda(dev_factors(tri, T, D, W, t0=t0), D)
    p0 = np.diff(np.concatenate([[0.0], F0]))
    p0 = np.maximum(p0, 0.0)

    idx = np.arange(lo_all, n_t)
    h = np.minimum(T - (idx + t0), D).astype(int)
    m = C[idx, h]

    nb = np.zeros((B, npool, D + 1), dtype=np.float64)
    for hv in np.unique(h):
        rows = np.where(h == hv)[0]
        pv = p0[: hv + 1]
        s = pv.sum()
        if s <= 0:
            nb[:, rows, 0] = m[rows]
            continue
        pv = pv / s
        counts = rng.multinomial(np.broadcast_to(m[rows].astype(np.int64),
                                                 (B, len(rows))), pv)
        nb[:, rows, : hv + 1] = counts
    Cb = np.cumsum(nb, axis=2)
    lam_b = _lambda_from_Cb(Cb, T, D, W, t0, lo_all)

    F = np.ones((B, D + 1), dtype=np.float64)
    for d in range(D - 1, -1, -1):
        F[:, d] = F[:, d + 1] / lam_b[:, d]
    return F
