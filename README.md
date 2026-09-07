# Reporting Delay and Delay-Corrected Nowcasting of NHTSA Vehicle Defect Complaints

Code and derived data for a study of reporting delay in the National Highway Traffic
Safety Administration (NHTSA) consumer complaint stream, and of a chain-ladder correction
that converts a partial incident-month count into an estimate of its eventual total.

Every current count of recent incidents in this stream is right truncated, because the
incidents have already happened while the complaints reporting them have not yet arrived.
This repository reconstructs the complaint stream at the record level, measures the delay
distribution on mature cohorts, corrects recent counts, and scores the correction against
realized outcomes at 66 pseudo-real-time origins.

## What is here

```
analysis/                  the full pipeline, ten numbered steps plus three modules
data/processed/            the analysis-ready complaint table (Parquet, 15 MB)
results/tables/            maturity factors, delay quantiles, hazard model, backtest inputs
results/audits/            per-step audit records, gate outcomes, and the backtest scoreboard
results/figures/           the four manuscript figures
CHECKSUMS.txt              SHA-256 of the raw NHTSA files behind data/processed/
```

## Reproducing the results

The analysis-ready table is included, so the statistical results can be reproduced without
downloading the raw files. Steps 2 through 10 read `data/processed/complaints_nowcast.parquet`
and rewrite everything under `results/`.

```bash
python -m venv .venv
.venv/Scripts/activate          # source .venv/bin/activate on Linux and macOS
pip install -r requirements.txt
python analysis/run_all.py
```

`run_all.py` runs every step in order and resets `results/audits/qa.md` first, so the gate
log always reflects exactly one clean run. Individual steps can be run on their own once
step 1 has produced the Parquet file.

To rebuild from the raw data instead, download the complaint flat files from the NHTSA
public file server, place them in `Data/` at the repository root, verify them against
`CHECKSUMS.txt`, and run step 1 first. The raw files total about 4.8 GB and are not
versioned here. NHTSA republishes them continuously, so a download made today will not
match the July 3, 2026 snapshot used for the published numbers. The checksums let you
confirm whether your copy is the same one.

## Pipeline

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `01_build_complaints.py` | record-level complaint table with incident and receipt dates |
| 2 | `02_delay_analysis.py` | delay quantiles, empirical curves, discrete-time reporting hazard |
| 3 | `03_triangle_nowcast.py` | reporting triangles, chain-ladder factors, unit tests |
| 4 | `04_backtest.py` | rolling-origin backtest, prediction intervals, coverage |
| 5 | `05_case_studies.py` | retrospective early-warning comparison |
| 6 | `06_figures.py` | manuscript figures |
| 7 | `07_tables.py` | manuscript tables |
| 8 | `08_robustness.py` | seven specification variants |
| 9 | `09_summary.py` | results summary |
| 10 | `10_qa_audit.py` | cross-step consistency checks |

`chainladder.py` holds the estimator and the censoring gate, `common.py` the shared
constants and paths, and `figstyle.py` the figure styling.

## Method in brief

Complaints are arranged in a reporting triangle indexed by incident month and reporting
delay in months. Development factors are estimated by the chain ladder on a rolling
60-month window of cohorts that are mature at the point of estimation, which yields a
maturity factor for each severity stream and each number of months elapsed. Dividing a
partial count by its maturity factor gives the nowcast.

Prediction intervals combine three sources of error. The first is sampling error in the
estimated delay distribution, drawn by a model-consistent multinomial bootstrap. The
second is the stochastic arrival of the complaints not yet received, treated as negative
binomial. The third is drift in the delay distribution between the estimation window and
the cohort being nowcast, whose scale is read at each origin from the gap between a
60-month and a 24-month fit of the same censored triangle.

Leakage discipline is enforced in one place. At each pseudo-cutoff the development
factors, the bootstrap, the dispersion, and the drift scale see only the output of a
single censoring function, and a direct test confirms that no cell later than the cutoff
survives it. The dispersion and the drift scale are fitted on origins through December
2020, and coverage is evaluated only on origins from January 2021.

## Data source

Complaint flat files published by the NHTSA Office of Defects Investigation, downloaded on
July 3, 2026. The files are United States government works in the public domain. Nothing
in this repository contains personally identifying information; the complaint narratives
are excluded and only the structured fields are retained.

## Licence

Code is released under the MIT Licence, in `LICENSE`. Derived data files under `data/` and
`results/` may be reused under CC BY 4.0 with attribution to the paper.

## Citation

Rafe, A., and Das, S. Reporting delay and delay-corrected nowcasting of vehicle defect
complaints. Manuscript under review. Please cite the paper when using this code or the
derived data.
