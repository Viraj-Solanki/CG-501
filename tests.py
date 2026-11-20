"""
tests.py -- Known-answer validation of the models.

Run with:   python tests.py        (no extra packages needed)
     or:    python -m pytest tests.py

Each test compares a model against an answer we can compute independently
(by hand or in closed form). If these pass, we can trust the simulations.
"""

import numpy as np

from models import KalmanLearner, RescorlaWagner, WithinCompoundRW, run
from schedules import backward_blocking, cue_vector


# ---------------------------------------------------------------------------
# Rescorla-Wagner
# ---------------------------------------------------------------------------
def test_rw_single_cue_converges_to_reward():
    """A+ trials: w_A must converge to the reward value r."""
    for r in [1.0, 0.7, -0.5]:
        model = RescorlaWagner(n_cues=1, alpha=0.1)
        hist = run(model, np.ones((300, 1)), np.full(300, r))
        assert abs(hist["w"][-1, 0] - r) < 1e-6, (r, hist["w"][-1, 0])


def test_rw_matches_closed_form_curve():
    """With one cue, RW has an exact solution: w_t = r * (1 - (1 - alpha)^t)."""
    alpha, r, T = 0.15, 1.0, 50
    hist = run(RescorlaWagner(1, alpha=alpha), np.ones((T, 1)), np.full(T, r))
    t = np.arange(1, T + 1)
    expected = r * (1 - (1 - alpha) ** t)
    assert np.allclose(hist["w"][:, 0], expected, atol=1e-12)


def test_rw_absent_cue_never_changes():
    """In RW an absent cue (x_i = 0) can never change its weight."""
    model = RescorlaWagner(3, alpha=0.2, w0=[0.3, 0.4, 0.5])
    hist = run(model, np.tile(cue_vector("AB"), (40, 1)), np.ones(40))
    assert np.all(hist["w"][:, 2] == 0.5)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
def ridge_posterior(X, R, sigma2, prior_var):
    """Closed-form Bayesian linear regression (= ridge) with prior N(0, prior_var I).

        precision = X'X / sigma2 + I / prior_var
        cov       = precision^-1
        mean      = cov @ X'R / sigma2
    """
    n = X.shape[1]
    precision = X.T @ X / sigma2 + np.eye(n) / prior_var
    cov = np.linalg.inv(precision)
    mean = cov @ X.T @ R / sigma2
    return mean, cov


def test_kalman_zero_process_noise_matches_ridge_random_data():
    """With tau2 = 0 the weights never drift, so the Kalman filter is just
    sequential Bayesian linear regression. After ALL trials its mean and
    covariance must equal the batch closed-form posterior -- and this holds
    after EVERY trial, not just the last, so we check every prefix."""
    rng = np.random.default_rng(0)                 # fixed seed
    T, n, sigma2, prior_var = 60, 3, 0.3, 2.0
    X = rng.integers(0, 2, size=(T, n)).astype(float)
    R = rng.normal(0.5, 1.0, size=T)

    model = KalmanLearner(n, sigma2=sigma2, tau2=0.0, prior_var=prior_var)
    hist = run(model, X, R)
    for t in range(1, T + 1):
        mean, cov = ridge_posterior(X[:t], R[:t], sigma2, prior_var)
        assert np.allclose(hist["w"][t - 1], mean, atol=1e-9)
        assert np.allclose(hist["S"][t - 1], cov, atol=1e-9)


def test_kalman_zero_process_noise_matches_ridge_backward_blocking():
    """Same check on the actual backward-blocking schedule used in the study."""
    sched, _ = backward_blocking(N=30)
    sigma2, prior_var = 0.5, 1.0
    model = KalmanLearner(3, sigma2=sigma2, tau2=0.0, prior_var=prior_var)
    hist = run(model, sched.X, sched.R)
    mean, cov = ridge_posterior(sched.X, sched.R, sigma2, prior_var)
    assert np.allclose(hist["w"][-1], mean, atol=1e-9)
    assert np.allclose(hist["S"][-1], cov, atol=1e-9)


def test_kalman_steady_state_variance_single_cue():
    """Tests the PROCESS-NOISE part (which the ridge test does not cover).
    For one always-present cue, the predicted variance P = S + tau2 settles
    where the update stops changing it. Solving
        P = tau2 + P*sigma2 / (P + sigma2)
    gives P = (tau2 + sqrt(tau2^2 + 4*tau2*sigma2)) / 2."""
    for tau2, sigma2 in [(0.01, 0.5), (0.1, 1.0), (0.001, 0.05)]:
        model = KalmanLearner(1, sigma2=sigma2, tau2=tau2, prior_var=1.0)
        hist = run(model, np.ones((2000, 1)), np.ones(2000))
        P_expected = (tau2 + np.sqrt(tau2 ** 2 + 4 * tau2 * sigma2)) / 2
        P_model = hist["S"][-1, 0, 0] + tau2
        assert abs(P_model - P_expected) < 1e-9, (tau2, sigma2, P_model, P_expected)


def test_kalman_covariance_stays_symmetric_positive_definite():
    sched, _ = backward_blocking(N=200)
    hist = run(KalmanLearner(3, sigma2=0.5, tau2=0.05), sched.X, sched.R)
    for S in hist["S"]:
        assert np.allclose(S, S.T)
        assert np.all(np.linalg.eigvalsh(S) > 0)


# ---------------------------------------------------------------------------
# Ablation and alternative model
# ---------------------------------------------------------------------------
def test_diagonal_ablation_keeps_absent_cue_fixed():
    """With the covariance ablated, an absent cue's gain is exactly 0."""
    sched, _ = backward_blocking(N=30)
    hist = run(KalmanLearner(3, diagonal=True), sched.X, sched.R)
    S_last = hist["S"][-1]
    assert np.all(S_last[~np.eye(3, dtype=bool)] == 0)   # off-diagonals zero
    phase2 = slice(30, 60)                                 # B+ trials, A absent
    assert np.all(hist["k"][phase2, 0] == 0)
    assert np.all(hist["w"][phase2, 0] == hist["w"][29, 0])


def test_diagonal_equals_full_when_only_one_cue_is_trained():
    """If cues are never presented together, no covariance is ever created,
    so the ablation should change nothing."""
    X = np.tile(cue_vector("A"), (50, 1))
    R = np.ones(50)
    full = run(KalmanLearner(3, tau2=0.02), X, R)
    diag = run(KalmanLearner(3, tau2=0.02, diagonal=True), X, R)
    assert np.allclose(full["w"], diag["w"])
    assert np.allclose(full["S"], diag["S"])


def test_within_compound_rw_reduces_to_rw():
    sched, _ = backward_blocking(N=30)
    rw = run(RescorlaWagner(3, alpha=0.1), sched.X, sched.R)
    wc = run(WithinCompoundRW(3, alpha=0.1, alpha_absent=0.0), sched.X, sched.R)
    assert np.allclose(rw["w"], wc["w"], atol=1e-12)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
