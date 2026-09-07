# QA log  -  gate results and fallbacks

Written by code (`common.qa_log`). One row per gate check.

| Step | Gate | Result | Detail |
|---|---|---|---|
| 1 | Gate 1  -  rows == 2,221,663 | PASS | observed 2,221,663 |
| 1 | Gate 1  -  unique ODINO == 1,603,201 | PASS | observed 1,603,201 |
| 1 | Gate 1  -  both-dates share ~95% | PASS | observed 0.9504 |
| 1 | Gate 1  -  negative delays ~582 | PASS | observed 582 |
| 1 | Gate 1  -  Y1 ~117k | PASS | observed 123,377 |
| 1 | Gate 1  -  Y2 ~84k | PASS | observed 86,566 |
| 1 | Gate 1  -  Y3 ~4.1k | PASS | observed 4,369 |
| 2 | Gate 2  -  overall median delay 2-3 weeks | PASS | 20 days |
| 2 | Gate 2  -  median(Y3) >= 4 x median(Y0) | PASS | Y3 99 d vs Y0 20 d (ratio 4.95) |
| 2 | Gate 2  -  outline's ~132 d Y3 median reproduces in its own variant | PASS | all-channels / no-1995-floor: Y3 132 d, overall 21 d; main spec (consumer+floor) Y3 99 d  -  difference documented in deviations.md |
| 3 | Gate 3  -  unit tests (i)-(iii) all pass | PASS | synthetic max|dF| = 0.00056; mature identity exact; F monotone with F(D)=1 |
| 3 | Gate 3  -  F_hat_Y3(3) < F_hat_Y0(3) | PASS | Y3 0.606 vs Y0 0.804 |
| 4 | Gate 4  -  |bias| nowcast < naive at h<=12 for Y2,Y3 | PASS | all pass |
| 4 | Gate 4  -  out-of-sample 90% coverage in [80%, 97%] (pooled per stream, n=150) | **FAIL** | pooled: Y0 0.99, Y1 0.92, Y2 0.97, Y3 0.91, all 0.99, severe 0.98; overall 0.960. Per-cell (n=30) range [0.83, 1.00]  -  at n=30 a perfectly calibrated 90% interval reads 1.00 ~4% of the time, so the gate is evaluated on the pooled rates |
| 5 | Gate 5  -  at least one primary case shows detection gain >= 1 month | PASS | Hyundai/Kia engine fires: 18; Chevrolet Bolt battery fires: neither alarms; ARC inflator (all makes, air bags): neither alarms |
| 5 | Gate 5  -  at least one primary case shows detection gain >= 1 month | PASS | Hyundai/Kia engine fires: 18; Chevrolet Bolt battery fires: neither alarms; ARC inflator (all makes, air bags): neither alarms |
| 5 | Gate 5  -  at least one primary case shows detection gain >= 1 month | PASS | Hyundai/Kia engine fires: 18; Chevrolet Bolt battery fires: neither alarms; ARC inflator (all makes, air bags): neither alarms |
| 4 | Gate 4  -  |bias| nowcast < naive at h<=12 for Y2,Y3 | PASS | all pass |
| 4 | Gate 4  -  out-of-sample 90% coverage in [80%, 97%] (pooled per stream, n=150) | **FAIL** | pooled: Y0 0.99, Y1 0.92, Y2 0.97, Y3 0.91, all 0.99, severe 0.98; overall 0.960. Per-cell (n=30) range [0.83, 1.00]  -  at n=30 a perfectly calibrated 90% interval reads 1.00 ~4% of the time, so the gate is evaluated on the pooled rates |
| 2 | Gate 2  -  overall median delay 2-3 weeks | PASS | 20 days |
| 2 | Gate 2  -  median(Y3) >= 4 x median(Y0) | PASS | Y3 99 d vs Y0 20 d (ratio 4.95) |
| 2 | Gate 2  -  outline's ~132 d Y3 median reproduces in its own variant | PASS | all-channels / no-1995-floor: Y3 132 d, overall 21 d; main spec (consumer+floor) Y3 99 d  -  difference documented in deviations.md |
| 4 | Gate 4  -  |bias| nowcast < naive at h<=12 for Y2,Y3 | PASS | all pass |
| 4 | Gate 4  -  out-of-sample 90% coverage in [80%, 97%] (pooled per stream, n=150) | **FAIL** | pooled: Y0 0.99, Y1 0.92, Y2 0.97, Y3 0.92, all 0.99, severe 0.99; overall 0.963. Per-cell (n=30) range [0.79, 1.00]  -  at n=30 a perfectly calibrated 90% interval reads 1.00 ~4% of the time, so the gate is evaluated on the pooled rates |
| 4 | Gate 4  -  |bias| nowcast < naive at h<=12 for Y2,Y3 | PASS | all pass |
| 4 | Gate 4  -  out-of-sample 90% coverage in [80%, 97%] (pooled per stream, n=150) | **FAIL** | pooled: Y0 0.99, Y1 0.91, Y2 0.99, Y3 0.92, all 0.99, severe 0.99; overall 0.964. Per-cell (n=30) range [0.79, 1.00]  -  at n=30 a perfectly calibrated 90% interval reads 1.00 ~4% of the time, so the gate is evaluated on the pooled rates |
| 4 | Gate 4  -  |bias| nowcast < naive at h<=12 for Y2,Y3 | PASS | all pass |
| 4 | Gate 4  -  out-of-sample 90% coverage in [80%, 97%] (pooled per stream, n=150) | **FAIL** | pooled: Y0 0.93, Y1 0.89, Y2 0.99, Y3 0.92, all 0.93, severe 0.97; overall 0.938. Per-cell (n=30) range [0.70, 1.00]  -  at n=30 a perfectly calibrated 90% interval reads 1.00 ~4% of the time, so the gate is evaluated on the pooled rates |
