"""
models.py -- Two associative-learning models for classical conditioning.

1. RescorlaWagner : the classic delta-rule learner (Rescorla & Wagner, 1972).
2. KalmanLearner  : a Bayesian learner that tracks a mean AND a covariance over
                    the associative weights (Dayan, Kakade & Montague, 2000;
                    Kakade & Dayan, 2002; see also Sutton, 1992; Kruschke, 2008).
3. WithinCompoundRW : RW plus cue-cue associations, a simplified version of the
                    Van Hamme & Wasserman (1994) idea. Used only as the
                    "alternative explanation" in the Discussion.

Both models share the same interface:
    model.predict(x)      -> predicted reward for stimulus vector x
    model.update(x, r)    -> learn from one trial (stimulus x, reward r)
and the helper `run(model, X, R)` runs a whole schedule of trials and
returns the trial-by-trial history.

Notation (used throughout the project)
--------------------------------------
    n        number of cues (e.g. n=2 for cues A and B)
    x_t      stimulus vector on trial t, shape (n,). x_i = 1 if cue i is
             present, 0 if absent.
    r_t      reward (US) on trial t (a scalar, e.g. 1 = food, 0 = nothing)
    w        associative weights, shape (n,). Prediction = x . w
    delta_t  prediction error = r_t - x_t . w

Both models are deterministic: given the same schedule they always give the
same result, so no random seed is needed here. Randomness (if any) lives in
schedules.py / experiments.py, where seeds are fixed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Rescorla-Wagner
# ---------------------------------------------------------------------------
class RescorlaWagner:
    """Rescorla-Wagner (1972) learner.

    Update rule (one trial):
        delta = r - sum_i x_i * w_i          (shared prediction error)
        w_i  <- w_i + alpha_i * x_i * delta  (only PRESENT cues change)

    Key properties:
      * The prediction error is computed from the SUM of all present cues,
        so cues compete for the same reward -> blocking, overshadowing.
      * x_i = 0 for an absent cue, so its weight cannot change.
        => RW predicts NO backward blocking and NO unovershadowing.
      * The learning rate alpha is fixed: it does not depend on how much
        the learner already knows.

    Parameters
    ----------
    n_cues : int
        Number of cues.
    alpha : float or array of shape (n_cues,)
        Learning rate(s). One value per cue lets you model cue salience
        (in the original paper this is alpha_i * beta; we fold both in).
    w0 : float or array
        Initial weights (default 0 = no prior association).
    """

    def __init__(self, n_cues, alpha=0.1, w0=0.0):
        self.n = n_cues
        # np.broadcast_to turns a single number into one value per cue.
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (n_cues,)).copy()
        self.w = np.broadcast_to(np.asarray(w0, dtype=float), (n_cues,)).copy()

    def predict(self, x):
        """Predicted reward = sum of the weights of the cues that are present."""
        return float(np.dot(x, self.w))

    def update(self, x, r):
        """Learn from one trial. Returns the prediction error."""
        x = np.asarray(x, dtype=float)
        delta = r - self.predict(x)            # how surprised were we?
        self.w = self.w + self.alpha * x * delta  # move present cues toward r
        return delta

    def state(self):
        """Snapshot used by `run` to record history."""
        return {"w": self.w.copy()}


# ---------------------------------------------------------------------------
# 2. Kalman-filter learner
# ---------------------------------------------------------------------------
class KalmanLearner:
    """Kalman-filter learner (Dayan, Kakade & Montague, 2000).

    Generative model (what the animal ASSUMES about the world):
        w_t = w_{t-1} + eta_t,    eta_t ~ N(0, tau2 * I)   (weights drift)
        r_t = x_t . w_t + eps_t,  eps_t ~ N(0, sigma2)     (noisy rewards)

    The learner keeps a Gaussian belief over the weights:
        w ~ N(mean = self.w, covariance = self.S)

    Update for one trial (standard Kalman filter equations):
        (1) Predict step -- the world may have drifted, so uncertainty grows:
                S_pred = S + tau2 * I
        (2) Kalman gain -- a per-cue, uncertainty-driven learning rate:
                k = S_pred x / (x' S_pred x + sigma2)
        (3) Mean update -- same shape as RW, but with learning rate k:
                delta = r - x . w
                w <- w + k * delta
        (4) Covariance update -- having seen data, we are more certain:
                S <- S_pred - k (x' S_pred)

    Why this differs from RW:
      * Learning rate k shrinks as uncertainty S shrinks (fast early,
        slow later) and grows again if tau2 > 0 (the world might change).
      * k_i = (S_pred x)_i = sum_j S_pred[i, j] x_j. For an ABSENT cue i
        (x_i = 0), k_i can still be non-zero if S_pred[i, j] != 0 for some
        present cue j. Training AB+ makes S[A, B] NEGATIVE (the data only
        tell us w_A + w_B ~ 1, so a high w_A implies a low w_B). Then
        training B+ alone raises w_B and, via the negative covariance,
        LOWERS w_A  ->  backward blocking. Training A- alone lowers w_A and
        RAISES w_B  ->  unovershadowing.

    Ablation (diagonal=True):
        After every update, the off-diagonal entries of S are set to zero,
        so the learner forgets how cues co-vary. Each cue still has its own
        uncertainty-driven learning rate, but k_i = 0 whenever x_i = 0.
        Prediction: backward blocking and unovershadowing should disappear,
        while forward blocking should survive (it comes from the shared
        prediction error, not from the covariance).

    Parameters
    ----------
    n_cues : int
        Number of cues.
    sigma2 : float
        Observation noise variance (how noisy rewards are believed to be).
    tau2 : float
        Process noise variance (how fast the weights are believed to drift).
        tau2 = 0 means the world is assumed stationary.
    prior_var : float
        Initial variance of each weight, S_0 = prior_var * I.
    w0 : float or array
        Initial weight means (default 0).
    diagonal : bool
        If True, ablate the cross-cue covariance (see above).
    """

    def __init__(self, n_cues, sigma2=0.5, tau2=0.01, prior_var=1.0,
                 w0=0.0, diagonal=False):
        self.n = n_cues
        self.sigma2 = float(sigma2)
        self.tau2 = float(tau2)
        self.diagonal = bool(diagonal)
        self.w = np.broadcast_to(np.asarray(w0, dtype=float), (n_cues,)).copy()
        self.S = prior_var * np.eye(n_cues)   # start: independent, uncertain
        self.last_gain = np.zeros(n_cues)     # stored for plotting

    def predict(self, x):
        """Predicted reward = x . (mean weights)."""
        return float(np.dot(x, self.w))

    def update(self, x, r):
        """Learn from one trial. Returns the prediction error."""
        x = np.asarray(x, dtype=float)

        # (1) Predict step: weights may have drifted since the last trial.
        S_pred = self.S + self.tau2 * np.eye(self.n)

        # (2) Kalman gain.
        Sx = S_pred @ x                       # vector, shape (n,)
        total_var = x @ Sx + self.sigma2      # variance of the predicted reward
        k = Sx / total_var                    # per-cue learning rate

        # (3) Mean update (same form as RW, but learning rate = k).
        delta = r - self.predict(x)
        self.w = self.w + k * delta

        # (4) Covariance update: S = S_pred - k (x' S_pred) = S_pred - k Sx'.
        S_new = S_pred - np.outer(k, Sx)
        # Rounding errors can make S very slightly non-symmetric; average it
        # with its transpose to keep it exactly symmetric.
        S_new = 0.5 * (S_new + S_new.T)

        # Ablation: throw away cross-cue covariance.
        if self.diagonal:
            S_new = np.diag(np.diag(S_new))

        self.S = S_new
        self.last_gain = k
        return delta

    def state(self):
        """Snapshot used by `run` to record history."""
        return {"w": self.w.copy(), "S": self.S.copy(), "k": self.last_gain.copy()}


# ---------------------------------------------------------------------------
# 3. Alternative explanation: RW with within-compound associations
# ---------------------------------------------------------------------------
class WithinCompoundRW:
    """RW extended so that ABSENT-but-expected cues can also learn.

    This is a simplified model in the spirit of Van Hamme & Wasserman (1994),
    who proposed that a cue which is absent but "retrieved" from memory
    (because it was previously paired with a present cue) has a NEGATIVE
    learning rate. It is the main non-Bayesian alternative explanation of
    backward blocking and unovershadowing. NOTE: this is OUR simplified
    implementation, not an exact reproduction of their published equations.

    Two kinds of learning:
      (a) Within-compound (cue-cue) associations a[i, j] = how strongly
          cue j calls cue i to mind. Learned with a delta rule whenever
          cue j is present:
              a[i, j] <- a[i, j] + lr_within * x_j * (x_i - a[i, j])
          (so a[i, j] grows on AB trials and decays on B-alone trials).
      (b) Cue-outcome weights, as in RW, but absent cues are activated by
          their retrieved strength c_i = min(1, sum_j a[i, j] x_j) and
          learn with a NEGATIVE rate alpha_absent:
              delta = r - x . w
              w_i <- w_i + alpha * x_i * delta          (present cues)
              w_i <- w_i + alpha_absent * c_i * delta   (absent cues)

    With alpha_absent = 0 this reduces exactly to plain Rescorla-Wagner.
    """

    def __init__(self, n_cues, alpha=0.1, alpha_absent=-0.05, lr_within=0.1, w0=0.0):
        self.n = n_cues
        self.alpha = float(alpha)
        self.alpha_absent = float(alpha_absent)
        self.lr_within = float(lr_within)
        self.w = np.broadcast_to(np.asarray(w0, dtype=float), (n_cues,)).copy()
        self.a = np.zeros((n_cues, n_cues))   # a[i, j]: cue j -> cue i

    def predict(self, x):
        return float(np.dot(x, self.w))

    def update(self, x, r):
        x = np.asarray(x, dtype=float)
        delta = r - self.predict(x)

        # Retrieved activation of ABSENT cues (present cues get 0 here).
        c = np.minimum(1.0, self.a @ x) * (1.0 - x)

        # Cue-outcome learning (computed from the OLD a, before (a) below).
        self.w = self.w + self.alpha * x * delta + self.alpha_absent * c * delta

        # Within-compound learning: column j updates only when cue j is present.
        self.a = self.a + self.lr_within * (x[:, None] - self.a) * x[None, :]
        np.fill_diagonal(self.a, 0.0)          # a cue does not predict itself
        return delta

    def state(self):
        return {"w": self.w.copy(), "a": self.a.copy()}


# ---------------------------------------------------------------------------
# Helper: run a model through a whole schedule
# ---------------------------------------------------------------------------
def run(model, X, R):
    """Run `model` on a sequence of trials.

    Parameters
    ----------
    model : RescorlaWagner or KalmanLearner (freshly constructed)
    X : array, shape (T, n_cues)  -- stimulus vector on each trial
    R : array, shape (T,)         -- reward on each trial

    Returns
    -------
    hist : dict of numpy arrays
        hist["w"][t]     weights AFTER trial t           shape (T, n)
        hist["pred"][t]  prediction made BEFORE trial t  shape (T,)
        hist["delta"][t] prediction error on trial t     shape (T,)
        For KalmanLearner also:
        hist["S"][t]     covariance AFTER trial t        shape (T, n, n)
        hist["k"][t]     Kalman gain used on trial t     shape (T, n)
    """
    X = np.asarray(X, dtype=float)
    R = np.asarray(R, dtype=float)
    records = {"pred": [], "delta": []}
    for x, r in zip(X, R):
        records["pred"].append(model.predict(x))
        records["delta"].append(model.update(x, r))
        for key, val in model.state().items():
            records.setdefault(key, []).append(val)
    return {key: np.array(val) for key, val in records.items()}
