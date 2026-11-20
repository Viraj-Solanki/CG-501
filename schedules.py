"""
schedules.py -- Trial sequences for the six conditioning paradigms.

A schedule is a list of PHASES. Each phase repeats one trial type n times,
e.g. ("AB", 1.0, 30) = 30 trials where cues A and B are shown together and
reward 1 is delivered. Conventional notation: "AB+" = AB followed by reward,
"A-" = A followed by no reward.

Every paradigm has a TEST schedule and a CONTROL schedule. The size of a
phenomenon is always measured as (test - control) on one critical cue, so
that changes caused by mere training length or time are subtracted out.

    Paradigm           Test                  Control           Critical cue
    -----------------  --------------------  ----------------  ------------
    acquisition        A+                    (none)            A
    extinction         A+ -> A-              (none)            A
    blocking           A+ -> AB+             C+ -> AB+         B
    overshadowing      AB+                   A+                A
    backward blocking  AB+ -> B+             AB+ -> C+         A
    unovershadowing    AB+ -> A-             AB+ -> C-         B

Controls use a third, unrelated cue C so that the number of trials and the
number of reinforced trials match the test condition. All schedules use the
same three-cue space [A, B, C] so every model has n_cues = 3.
"""

from dataclasses import dataclass, field

import numpy as np

CUES = ["A", "B", "C"]


@dataclass
class Schedule:
    """A trial sequence ready to feed into models.run(model, X, R)."""
    name: str
    X: np.ndarray                 # (T, 3) stimulus vectors
    R: np.ndarray                 # (T,)   rewards
    phases: list = field(default_factory=list)   # [(label, start, end), ...]

    @property
    def n_trials(self):
        return len(self.R)


def cue_vector(trial_type):
    """'AB' -> array([1, 1, 0])."""
    return np.array([1.0 if c in trial_type else 0.0 for c in CUES])


def make_schedule(name, phases):
    """Build a Schedule from a list of (trial_type, reward, n_trials)."""
    X, R, phase_info, t = [], [], [], 0
    for trial_type, reward, n in phases:
        X += [cue_vector(trial_type)] * n
        R += [float(reward)] * n
        sign = "+" if reward > 0 else "-"
        phase_info.append((f"{trial_type}{sign}", t, t + n))
        t += n
    return Schedule(name, np.array(X), np.array(R), phase_info)


def add_reward_noise(schedule, sd, seed):
    """Return a copy of `schedule` with Gaussian noise added to every reward.

    Used for the robustness check: real outcomes are never perfectly
    identical across trials. The seed makes it exactly reproducible.
    """
    rng = np.random.default_rng(seed)
    R_noisy = schedule.R + rng.normal(0.0, sd, size=schedule.R.shape)
    return Schedule(schedule.name, schedule.X.copy(), R_noisy, list(schedule.phases))


# ---------------------------------------------------------------------------
# The six paradigms. N = trials per phase.
# ---------------------------------------------------------------------------
def acquisition(N=30):
    return make_schedule("acquisition", [("A", 1, N)])


def extinction(N=30):
    return make_schedule("extinction", [("A", 1, N), ("A", 0, N)])


def blocking(N=30):
    test = make_schedule("blocking", [("A", 1, N), ("AB", 1, N)])
    control = make_schedule("blocking_control", [("C", 1, N), ("AB", 1, N)])
    return test, control


def overshadowing(N=30):
    test = make_schedule("overshadowing", [("AB", 1, N)])
    control = make_schedule("overshadowing_control", [("A", 1, N)])
    return test, control


def backward_blocking(N=30, N1=None):
    """N1 = length of the AB+ phase (defaults to N)."""
    N1 = N if N1 is None else N1
    test = make_schedule("backward_blocking", [("AB", 1, N1), ("B", 1, N)])
    control = make_schedule("backward_blocking_control", [("AB", 1, N1), ("C", 1, N)])
    return test, control


def unovershadowing(N=30, N1=None):
    N1 = N if N1 is None else N1
    test = make_schedule("unovershadowing", [("AB", 1, N1), ("A", 0, N)])
    control = make_schedule("unovershadowing_control", [("AB", 1, N1), ("C", 0, N)])
    return test, control


# Paradigms that have a test/control pair, with the critical cue index and
# the sign that makes "phenomenon present" a POSITIVE number.
#   effect = sign * (w_test[cue] - w_control[cue])   at the end of training
PAIRED = {
    "blocking":          (blocking,          1, -1),  # B lower than control
    "overshadowing":     (overshadowing,     0, -1),  # A lower than A-alone
    "backward_blocking": (backward_blocking, 0, -1),  # A lower than control
    "unovershadowing":   (unovershadowing,   1, +1),  # B higher than control
}
