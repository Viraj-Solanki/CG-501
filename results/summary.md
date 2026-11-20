# Summary of results

Default parameters: {"N_TRIALS": 30, "RW": {"alpha": 0.1}, "Kalman": {"sigma2": 0.5, "tau2": 0.01, "prior_var": 1.0}, "RW+within": {"alpha": 0.1, "alpha_absent": -0.05, "lr_within": 0.1}, "N_SEEDS": 20, "REWARD_NOISE_SD": 0.2}

## Exp 0 -- effect sizes, deterministic rewards (positive = phenomenon present)

| Model | blocking | overshadowing | backward blocking | unovershadowing | acq. trials to 90% | ext. trials to 50% | w_A after extinction |
|---|---|---|---|---|---|---|---|
| RW | 0.478 | 0.458 | 0.000 | 0.000 | 22 | 7 | 0.041 |
| Kalman | 0.498 | 0.498 | 0.419 | 0.419 | 5 | 5 | 0.014 |
| Kalman-diag | 0.498 | 0.498 | 0.000 | 0.000 | 5 | 5 | 0.014 |
| RW+within | 0.478 | 0.458 | 0.126 | 0.126 | 22 | 7 | 0.041 |

## Exp 3 -- robustness: mean +/- SD over 20 seeds, reward noise SD 0.2 (n seeds with effect > 0)

| Model | blocking | overshadowing | backward blocking | unovershadowing |
|---|---|---|---|---|
| RW | 0.464 +/- 0.023 (20/20) | 0.454 +/- 0.021 (20/20) | 0.000 +/- 0.000 (0/20) | 0.000 +/- 0.000 (0/20) |
| Kalman | 0.469 +/- 0.052 (20/20) | 0.485 +/- 0.024 (20/20) | 0.433 +/- 0.047 (20/20) | 0.405 +/- 0.047 (20/20) |
| Kalman-diag | 0.470 +/- 0.056 (20/20) | 0.490 +/- 0.024 (20/20) | 0.000 +/- 0.000 (0/20) | 0.000 +/- 0.000 (0/20) |
| RW+within | 0.464 +/- 0.023 (20/20) | 0.454 +/- 0.021 (20/20) | 0.131 +/- 0.013 (20/20) | 0.121 +/- 0.013 (20/20) |

## Exp 1 -- sweep ranges (Kalman, full covariance)

- BB effect (30 B+ trials): min 0.230, max 0.500
- BB effect (5 B+ trials): min 0.095, max 0.498
- BB at tau2 = 0 (30 B+ trials): min 0.271, max 0.500
- UO effect: min 0.199, max 0.499
- FB effect: min 0.271, max 0.500
- Kalman-diag BB: max |effect| 0.00e+00
- Kalman-diag UO: max |effect| 0.00e+00
- Kalman-diag FB: min 0.310, max 0.500
- Acquisition trials to 90%: min 1, max 90

## Exp 0b -- Dayan & Kakade (2001) Fig 2C-D replication (params {'sigma2': 0.12249999999999998, 'tau2': 0.0081, 'prior_var': 0.0081})

- w after 20 AB+: A 0.500, B 0.500
- w after 20 A+: A 0.999, B 0.208
- S_AA at trial 20 0.094, at end 0.028
- S_AB at trial 20 -0.076, at end -0.0002
- Kalman-diag w_B at end: 0.500

## Exp 2d -- why BB is non-monotonic in tau2 (sigma2 = 0.5)

| tau2 | corr(A,B) entering B+ | k_A on 1st B+ trial | w_A after 1 B+ | after 5 | after 30 |
|---|---|---|---|---|---|
| 0.0 | -0.984 | -0.494 | 0.247 | 0.082 | 0.016 |
| 0.001 | -0.968 | -0.495 | 0.249 | 0.090 | 0.034 |
| 0.01 | -0.919 | -0.530 | 0.235 | 0.105 | 0.081 |
| 0.03 | -0.899 | -0.602 | 0.199 | 0.102 | 0.094 |
| 0.1 | -0.900 | -0.731 | 0.135 | 0.085 | 0.084 |
| 1.0 | -0.927 | -0.900 | 0.050 | 0.046 | 0.046 |

## Exp 2e -- trial order: same trials, forward (A+ -> AB+) vs backward (AB+ -> A+) order; final w_B (lower = more blocking)

| Model | w_B forward order | w_B backward order |
|---|---|---|
| Kalman tau2=0 | 0.016 | 0.016 |
| Kalman tau2=0.01 | 0.002 | 0.081 |
| Kalman tau2=0.1 | 0.000 | 0.084 |
| RW+within | 0.021 | 0.373 |
| RW | 0.021 | 0.499 |
