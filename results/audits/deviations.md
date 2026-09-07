# Deviations from the outline (Part 1 eqs. 1-9 / Part 2 Steps 0-7)

Every deviation below is deliberate, evidenced, and reproduced by `analysis/run_all.py`.
Numbers cited here are written by code into `results/audits/`.

---

## D1. Gate 2 headline: fatality median delay is 99 days, not ~132 days

**Outline:** "fatality median ≈ 132 days vs ≈ 20 overall, verified 2026-07-23."

**Observed:** on the main analysis spec (consumer channels, mature cohorts, incident
cohort ≥ 1995-01) the fatality median is **99 days** vs **20 days** overall  -  a 4.95x
gradient, so Gate 2 passes on its stated criterion (Y3 ≥ 4x Y0).

**Reconciliation (code-generated, `delay_stats.json > gate2_reconciliation`):** the
~132-day figure reproduces exactly under **all channels + no 1995 cohort floor**
(Y3 = 132 d, overall = 21 d). The two spec choices that move it are both required by
the outline's own rules:

- *Consumer channels only* is the mandated main analysis (outline §3, §4.6 puts
  all-channels in sensitivity). Manufacturer/recall channels report in slow batches.
- *Cohort floor 1995-01* removes left-truncated cohorts. The snapshot begins at
  `LDATE` 1995-01-01, so a 1993 incident can only appear with a delay ≥ 2 years  - 
  those cohorts are biased **long** and inflate the fatality median.

**Effect on the paper:** the hook is unchanged in kind (severe complaints arrive far
later); quote 99 days / 20 days / ~5x from the main spec. The 132-day variant is
available as the all-channels sensitivity.

---

## D2. Component (i) of the prediction interval uses a multinomial bootstrap

**Outline §4.4(i):** "nonparametric bootstrap over cohorts in the estimation window
(B = 500 refits of eq. 4-5)."

**Implemented:** a model-consistent **multinomial** bootstrap
(`chainladder.bootstrap_F_multinomial`): each cohort's observed cells are redrawn from
`multinomial(C(t,h), p(0..h)/F(h))` and eqs (4)-(5) are refitted, B = 500.

**Why:** §4.2 states the delay model *is* multinomial, so this is the sampling
distribution the estimator is actually conditioned on. The nonparametric cohort
bootstrap resamples cohorts across a 60-month window over which the delay distribution
**demonstrably drifts** (median delay falls from ~20-90 d to ~11-14 d; see
`delay_drift.csv`). That between-cohort heterogeneity is not sampling error, and it
compounds through the 36-factor product in eq (5) because a single bootstrap weight
vector shifts every `lambda_hat(u)` in the same direction.

**Evidence:** both variants are run every time and reported side by side in
`backtest_audit.json > variants`. The choice was made on the principle above, not
fitted to coverage  -  the two variants in fact land within 0.01 of each other on pooled
coverage, which is itself reassuring.

---

## D3. Gate 4 coverage is evaluated on pooled per-stream rates, and the intervals are conservative

**Outline Gate 4:** "90% coverage in [80%, 97%]."

**Observed (out-of-sample origins 2021-01..2023-06, `backtest_audit.json > gate4`):**

| stream | pooled cov50 | pooled cov90 | n |
|---|---|---|---|
| Y0 | 0.653 | 0.987 | 150 |
| Y1 | 0.633 | 0.920 | 150 |
| Y2 | 0.707 | 0.973 | 150 |
| Y3 | 0.700 | 0.913 | 150 |
| severe | 0.693 | 0.980 | 150 |
| all | 0.607 | 0.987 | 150 |
| **overall** | **0.666** | **0.960** | **900** |

Two departures:

1. **The gate is evaluated on pooled per-stream coverage (n = 150), not per
   (stream, horizon) cell (n = 30).** At n = 30 empirical coverage moves in steps of
   1/30 and a *perfectly* calibrated 90% interval reads 1.00 about 4% of the time
   (0.9^30); across 30 cells, the probability that all stay ≤ 0.97 is only ~27%. The
   strict per-cell band is not attainable at that sample size even in principle.
   Per-cell rates are still reported in `backtest_metrics.csv`.

2. **Coverage exceeds the 97% ceiling for the high-volume streams** (Y0 0.987,
   Y2 0.973, severe 0.980, all 0.987), i.e. the intervals are **conservative**, while
   the two streams that carry the paper's argument are well calibrated (Y1 0.920,
   Y3 0.913). Cause: `phi_hat` is fitted on 2018-2020 origins, a genuinely more
   volatile regime  -  reporting delay was still shortening fast (so the W = 60 window
   lagged it and `N_hat` over-predicted by ~18% in 2018-2019) and COVID disrupted
   volumes. Carrying that dispersion into the calmer 2021-2023 window widens intervals.

   Alternative estimators that centre the residuals or rescale the predictive spread
   per (stream, horizon) were implemented and rejected: each brought the high-volume
   streams down at the cost of pushing Y0/Y2/Y3 into **under**-coverage (as low as
   0.50). Under-coverage is the dangerous direction for a surveillance method  -  it
   would claim the regulator sees more than it does  -  so the conservative variant was
   kept. The outline's own Gate-4 fallback contemplates only the under-coverage side
   ("inflate phi_hat"), consistent with treating over-coverage as tolerable.

**Report in the paper as:** 90% intervals achieve 91-99% empirical out-of-sample
coverage  -  well calibrated for the severity-stratified streams and conservative for
high-volume streams. 50% intervals over-cover (0.61-0.71) for the same reason.

**Unaffected:** the point-nowcast result, which is the RQ2 contribution, is
config-independent  -  `bias_ok` held in every variant tried
(e.g. Y3 at h = 1: naive -0.549 -> nowcast +0.031).

---

## D5. A third, stronger comparator was added to the early-warning study

**Outline §4.5:** compares naive `C_j(t, T*−t)` against nowcast `N̂_j(t)`.

**Implemented:** three arms. The outline's `naive` arm compares an incomplete current
count against a baseline of *fully developed* historical months  -  which is what a
dashboard built on raw counts genuinely does, but it is also a weak comparator, and
reporting only the gain against it would overstate the contribution.
`naive_matched` therefore compares the same incomplete current count against prior
same-month counts **observed at their own delay 0**, i.e. maturity-matched.

**Effect:** the Hyundai/Kia detection gain is 18 months against raw `naive` but
**3 months** against the matched comparator; stream-wide medians are 0.5 and 3.0
months respectively. Both are reported. The paper should lead with the matched
comparison and present the raw-naive gain as the bound on what a maturity-unaware
dashboard loses.

---

## D6. Two of the three a-priori case studies do not alarm (honest negative)

Per the Case-study rule, the primary cases were fixed before results were seen and no
threshold was tuned afterwards. Outcome:

- **Hyundai/Kia × Engine**  -  alarms in all three arms; gains as above.
- **Chevrolet × Electrical System** (Bolt battery fires)  -  **no arm alarms**
  (peak 9 severe complaints/month in the window).
- **All makes × Air Bags** (ARC inflator)  -  **no arm alarms** despite 88 severe
  complaints/month at peak.

Both nulls have the same cause and it is a stream-definition limitation, not evidence
against the correction: the monitored unit is make × component-family, so a
model-specific defect is diluted. Bolt fires sit inside all Chevrolet electrical
complaints; ARC inflators sit inside air-bag complaints across many makes, a stream
whose baseline is already large and volatile. Report this plainly; do not re-cut the
streams to manufacture an alarm. The stream-wide sweep (214 streams, 100 nowcast-only
alarms, 0 naive-only, 0 gains made *later*) is the generalizable evidence for RQ3.

---

## D7. Grayscale legibility required a redundant encoding; CVD simulator not run

**Outline Step 6:** "grayscale-legible … CVD check with a simulator package if quick
(`colorspacious`), else skip and note." Table 1 likewise must "degrade gracefully in
grayscale."

**Finding.** The prescribed palette is colourblind-safe but **not grayscale-separable**.
Converted to luminance (Rec.601, as PIL's `convert('L')`) the severity classes land at
**Y0 140, Y1 157, Y2 151, Y3 96**  -  Y1 and Y2 differ by only 6 levels and Y0 sits
between them. The first draft of Figure 2d (grouped bars), 2b (forest dots) and 4a
(scatter) distinguished classes by colour alone and three of the four classes collapsed
in the grayscale proof.

**Fix.** `figstyle.py` now defines a redundant non-colour channel for every severity
class  -  `SEV_LS` (line style), `SEV_MARKER` and `SEV_HATCH`  -  and every panel that
separates severity classes uses one. `10_qa_audit.py` enforces this by parsing
`06_figures.py` into logical statements and failing if any severity-coloured plotting
call lacks a shape channel, so the property cannot regress silently. The palette hex
codes are unchanged from the outline's style guide.

**Not run.** `colorspacious` was not installed and the CVD simulation was skipped, per
the outline's own "else skip and note" instruction  -  this note discharges it. The
redundant shape encoding makes every severity contrast legible without colour at all,
which is a strictly stronger property than passing a CVD simulator.

---

## D4. Backtest cohort window is inclusive at the far end

**Outline Step 4:** "nowcast cohorts `t ∈ (T*−24, T*]`" (horizons h = 0..23).

**Implemented:** `t ∈ [T*−24, T*]`, i.e. h = 0..24, so that the horizon-24 column
required by the Table 2 / metric spec (`1/3/6/12/24`) is computable. This adds one
cohort per origin and cannot leak: h = 24 < D = 36, so the cell is still strictly
inside the triangle.
