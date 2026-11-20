# Kalman-filter vs Rescorla-Wagner learners of classical conditioning

CG-501 (Computation and Cognition) end-term project, Solanki Viraj Rajeshbhai
(24110348).

**Deliverables:** report: [`report/report.pdf`](report/report.pdf) ·
slides: [`slides/slides.pdf`](slides/slides.pdf) ·
video (presentation + code walkthrough, 17 min): [Google Drive](https://drive.google.com/file/d/1Sk6oZtl0JrjcowHi4mo9hXpt0RBhHCt8/view?usp=sharing)

**Question.** Can a Kalman-filter learner with uncertainty-driven learning rates
explain conditioning phenomena that Rescorla-Wagner (RW) cannot (backward
blocking, unovershadowing), and which component of the Kalman filter is
responsible?

## Reproduce everything

Requirements: Python 3.9+, NumPy, Matplotlib (`pip install -r requirements.txt`) (tested with Python 3.12,
NumPy 2.5, Matplotlib 3.11). SciPy is not needed.

```bash
python tests.py          # known-answer validation (10 tests)
python make_figures.py   # runs all experiments, writes results/ and figures/
```

Every simulation is deterministic. The only randomness is the reward-noise
robustness check, which uses fixed seeds 0-19, so reruns give identical output.

## Files

| File | Contents |
|---|---|
| `models.py` | `RescorlaWagner`, `KalmanLearner` (with `diagonal=True` ablation), `WithinCompoundRW` (alternative model), `run()` |
| `schedules.py` | Test and control schedules for acquisition, extinction, blocking, overshadowing, backward blocking, unovershadowing |
| `experiments.py` | Exp 0 replication (+ 0b Dayan & Kakade 2001 replication), Exp 1 tau2 x sigma2 sweep, Exp 2 ablation (incl. trial-order test), Exp 3 reward-noise robustness |
| `make_figures.py` | Runs everything and draws figures 1-7 |
| `tests.py` | RW convergence and closed-form curve; Kalman (tau2 = 0) equals the Bayesian ridge posterior after every trial; Kalman steady-state variance; ablation sanity checks |
| `results/summary.md` | All numbers quoted in the report |
| `figures/` | fig1 replication, fig2 effect sizes, fig3 Kalman mechanism, fig4 noise sweep, fig5 ablation, fig6 alternative model, fig7 Dayan & Kakade (2001) replication |
| `slides/` | `slides.tex` / `slides.pdf` (Beamer; build the same way as the report) |
| `report/` | `report.tex` / `report.pdf`, IEEE conference format (`IEEEtran.cls` included; compile with `pdflatex report.tex` twice, from inside `report/`) |
| `requirements.txt` | Python packages needed (`pip install -r requirements.txt`) |

## Default parameters

RW alpha = 0.1. Kalman: sigma2 = 0.5, tau2 = 0.01, prior variance 1.
RW + within-compound: alpha = 0.1, alpha_absent = -0.05, lr_within = 0.1.
30 trials per phase. Effects are measured as test minus control on the
critical cue (see `schedules.py`).
