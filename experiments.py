"""
experiments.py -- All simulations for the project.

    Exp 0  Replication : 6 paradigms x 4 models (RW, Kalman, Kalman-diag,
                         RW+within-compound). Trajectories + effect sizes.
    Exp 1  Noise sweep : grid over process noise tau2 and observation noise
                         sigma2. Measures backward-blocking (BB),
                         unovershadowing (UO), forward-blocking (FB) size and
                         acquisition speed, for the full and ablated Kalman.
    Exp 2  Ablation    : full vs diagonal covariance; BB as a function of
                         tau2 and of the length of the AB+ phase.
    Exp 3  Robustness  : 20 fixed seeds of Gaussian reward noise; mean, SD
                         and sign-consistency of every effect for every model.

Running this file directly saves results/results.json and prints a summary.
make_figures.py imports run_all() so one command regenerates everything.
"""

import json
import os

import numpy as np

from models import KalmanLearner, RescorlaWagner, WithinCompoundRW, run
import schedules as sch

# ---------------------------------------------------------------------------
# Default parameters (used everywhere unless a sweep overrides them)
# ---------------------------------------------------------------------------
N_TRIALS = 30                     # trials per phase
RW_PARAMS = dict(alpha=0.1)
KF_PARAMS = dict(sigma2=0.5, tau2=0.01, prior_var=1.0)
WC_PARAMS = dict(alpha=0.1, alpha_absent=-0.05, lr_within=0.1)

N_SEEDS = 20                      # robustness check
REWARD_NOISE_SD = 0.2

# Sweep grids. tau2 = 0 is included as a separate first column.
TAU2_GRID = np.concatenate([[0.0], np.logspace(-4, 0, 9)])
SIGMA2_GRID = np.logspace(-2, 1, 10)

MODEL_NAMES = ["RW", "Kalman", "Kalman-diag", "RW+within"]


def make_model(name, **override):
    """Fresh model by name, with optional parameter overrides."""
    if name == "RW":
        return RescorlaWagner(3, **{**RW_PARAMS, **override})
    if name == "Kalman":
        return KalmanLearner(3, **{**KF_PARAMS, **override})
    if name == "Kalman-diag":
        return KalmanLearner(3, diagonal=True, **{**KF_PARAMS, **override})
    if name == "RW+within":
        return WithinCompoundRW(3, **{**WC_PARAMS, **override})
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def paired_effect(model_name, paradigm, N=N_TRIALS, noise_seed=None, **override):
    """Effect size for a test/control paradigm (positive = phenomenon present).

    effect = sign * (w_test[cue] - w_control[cue]) at the end of training.

    With noise_seed set, test and control receive the SAME noise sequence
    (matched design / "common random numbers"): trial t gets the same noise
    in both, so the groups differ only in the experimental manipulation.
    With independent seeds, the shared AB+ phase would end at slightly
    different weights in the two groups, and that difference would show up as
    a spurious "effect" even in RW, which provably has none.
    """
    builder, cue, sign = sch.PAIRED[paradigm]
    test, control = builder(N)
    if noise_seed is not None:
        test = sch.add_reward_noise(test, REWARD_NOISE_SD, noise_seed)
        control = sch.add_reward_noise(control, REWARD_NOISE_SD, noise_seed)
    h_test = run(make_model(model_name, **override), test.X, test.R)
    h_ctrl = run(make_model(model_name, **override), control.X, control.R)
    effect = sign * (h_test["w"][-1, cue] - h_ctrl["w"][-1, cue])
    return float(effect), h_test, h_ctrl


def bb_early(model_name, n_phase2=5, **override):
    """Backward blocking after only `n_phase2` B+ trials (instead of 30).
    The end-of-training measure saturates (w_A can only fall from ~0.5 to 0),
    so this early measure shows parameter differences more clearly."""
    test, control = sch.backward_blocking(N=n_phase2, N1=N_TRIALS)
    h_t = run(make_model(model_name, **override), test.X, test.R)
    h_c = run(make_model(model_name, **override), control.X, control.R)
    return float(h_c["w"][-1, 0] - h_t["w"][-1, 0])


def trials_to_criterion(values, criterion):
    """First trial (1-based) on which `values` reaches `criterion`; NaN if never."""
    idx = np.nonzero(values >= criterion)[0]
    return float(idx[0] + 1) if len(idx) else float("nan")


def acquisition_speed(model_name, N=100, **override):
    """Trials needed for w_A to reach 90% of the reward value (A+ trials)."""
    s = sch.acquisition(N)
    h = run(make_model(model_name, **override), s.X, s.R)
    return trials_to_criterion(h["w"][:, 0], 0.9)


def extinction_speed(model_name, N=N_TRIALS, **override):
    """Trials of A- needed for w_A to fall below half its pre-extinction value."""
    s = sch.extinction(N)
    h = run(make_model(model_name, **override), s.X, s.R)
    w_pre = h["w"][N - 1, 0]
    return trials_to_criterion(-h["w"][N:, 0], -0.5 * w_pre), float(h["w"][-1, 0])


# ---------------------------------------------------------------------------
# Exp 0: replication
# ---------------------------------------------------------------------------
def exp_replication():
    out = {"trajectories": {}, "effects": {}, "acq_trials": {},
           "ext_trials": {}, "ext_final": {}}
    for m in MODEL_NAMES:
        traj = {}
        s = sch.acquisition(N_TRIALS)
        traj["acquisition"] = {"test": run(make_model(m), s.X, s.R)["w"].tolist()}
        s = sch.extinction(N_TRIALS)
        traj["extinction"] = {"test": run(make_model(m), s.X, s.R)["w"].tolist()}
        out["effects"][m] = {}
        for p in sch.PAIRED:
            eff, h_t, h_c = paired_effect(m, p)
            out["effects"][m][p] = eff
            traj[p] = {"test": h_t["w"].tolist(), "control": h_c["w"].tolist()}
            if m == "Kalman" and p == "backward_blocking":
                # Keep the Kalman internals for the mechanism figure.
                out["bb_kalman_S"] = h_t["S"].tolist()
                out["bb_kalman_k"] = h_t["k"].tolist()
        out["trajectories"][m] = traj
        out["acq_trials"][m] = acquisition_speed(m)
        ext_t, ext_final = extinction_speed(m)
        out["ext_trials"][m] = ext_t
        out["ext_final"][m] = ext_final
    return out


# ---------------------------------------------------------------------------
# Exp 0b: replication of Dayan & Kakade (2001), Figure 2C-D
# ---------------------------------------------------------------------------
# Their simulation: 20 trials of x = (1,1) then 20 trials of x = (1,0), r = 1,
# exact Kalman filter with drift SD 0.09 and observation-noise SD 0.35.
# NOTE their notation is swapped relative to ours: THEIR sigma^2 is the drift
# (our tau2) and THEIR tau^2 is the observation noise (our sigma2).
# Their initial covariance Sigma_0 is not stated; we assume Sigma_0 = drift
# variance * I, which reproduces the scale of their covariance plot.
# Here A plays the role of their light L (trained alone in phase 2) and B the
# role of their sound S (the explained-away cue).
DK_PARAMS = dict(sigma2=0.35 ** 2, tau2=0.09 ** 2, prior_var=0.09 ** 2)


def exp_dk_replication():
    s = sch.make_schedule("dk2001", [("AB", 1, 20), ("A", 1, 20)])
    out = {}
    for m in ["Kalman", "Kalman-diag", "RW"]:
        over = DK_PARAMS if m != "RW" else {}
        h = run(make_model(m, **over), s.X, s.R)
        out[m] = {"w": h["w"][:, :2].tolist()}
        if "S" in h:
            out[m]["S_AA"] = h["S"][:, 0, 0].tolist()
            out[m]["S_AB"] = h["S"][:, 0, 1].tolist()
    return out


# ---------------------------------------------------------------------------
# Exp 1: tau2 x sigma2 sweep
# ---------------------------------------------------------------------------
def exp_noise_sweep():
    shape = (len(SIGMA2_GRID), len(TAU2_GRID))
    res = {k: np.zeros(shape) for k in
           ["bb_full", "bb_diag", "bb_early", "uo_full", "uo_diag",
            "fb_full", "fb_diag", "acq"]}
    for i, s2 in enumerate(SIGMA2_GRID):
        for j, t2 in enumerate(TAU2_GRID):
            p = dict(sigma2=s2, tau2=t2)
            res["bb_full"][i, j] = paired_effect("Kalman", "backward_blocking", **p)[0]
            res["bb_diag"][i, j] = paired_effect("Kalman-diag", "backward_blocking", **p)[0]
            res["bb_early"][i, j] = bb_early("Kalman", **p)
            res["uo_full"][i, j] = paired_effect("Kalman", "unovershadowing", **p)[0]
            res["uo_diag"][i, j] = paired_effect("Kalman-diag", "unovershadowing", **p)[0]
            res["fb_full"][i, j] = paired_effect("Kalman", "blocking", **p)[0]
            res["fb_diag"][i, j] = paired_effect("Kalman-diag", "blocking", **p)[0]
            res["acq"][i, j] = acquisition_speed("Kalman", **p)
    out = {k: v.tolist() for k, v in res.items()}
    out["tau2"] = TAU2_GRID.tolist()
    out["sigma2"] = SIGMA2_GRID.tolist()
    return out


# ---------------------------------------------------------------------------
# Exp 2: ablation
# ---------------------------------------------------------------------------
def exp_ablation():
    out = {}
    # (a) BB vs tau2 at default sigma2, full vs diag (and RW for reference).
    tau2_fine = np.concatenate([[0.0], np.logspace(-4, 0, 25)])
    out["tau2_fine"] = tau2_fine.tolist()
    for m in ["Kalman", "Kalman-diag"]:
        out[f"bb_vs_tau2_{m}"] = [paired_effect(m, "backward_blocking", tau2=t)[0]
                                  for t in tau2_fine]
    # (b) BB vs length of the AB+ phase, for several tau2 values.
    N1_values = np.arange(1, 101)
    out["N1_values"] = N1_values.tolist()
    out["bb_vs_N1"] = {}
    for t2 in [0.0, 0.001, 0.01, 0.1]:
        effs = []
        for N1 in N1_values:
            test, ctrl = sch.backward_blocking(N=N_TRIALS, N1=int(N1))
            ht = run(make_model("Kalman", tau2=t2), test.X, test.R)
            hc = run(make_model("Kalman", tau2=t2), ctrl.X, ctrl.R)
            effs.append(float(hc["w"][-1, 0] - ht["w"][-1, 0]))
        out["bb_vs_N1"][str(t2)] = effs
    # (c) The covariance S[A,B] at the end of the AB+ phase, same tau2 values.
    out["cov_AB_vs_N1"] = {}
    for t2 in [0.0, 0.001, 0.01, 0.1]:
        s = sch.make_schedule("ab", [("AB", 1, int(N1_values[-1]))])
        h = run(make_model("Kalman", tau2=t2), s.X, s.R)
        out["cov_AB_vs_N1"][str(t2)] = h["S"][:, 0, 1].tolist()
    # (e) Trial-order test: the SAME trials (30 A+, 30 AB+) in forward order
    # (A+ -> AB+) vs backward order (AB+ -> A+); final w_B in each. With tau2 = 0
    # the Bayesian posterior depends only on the set of trials, so the two
    # orders must give the same answer; drift (tau2 > 0) breaks the symmetry.
    fwd = sch.make_schedule("fwd", [("A", 1, N_TRIALS), ("AB", 1, N_TRIALS)])
    bwd = sch.make_schedule("bwd", [("AB", 1, N_TRIALS), ("A", 1, N_TRIALS)])
    out["order_test"] = []
    for label, m, over in [("Kalman tau2=0", "Kalman", dict(tau2=0.0)),
                           ("Kalman tau2=0.01", "Kalman", dict(tau2=0.01)),
                           ("Kalman tau2=0.1", "Kalman", dict(tau2=0.1)),
                           ("RW+within", "RW+within", {}), ("RW", "RW", {})]:
        wf = run(make_model(m, **over), fwd.X, fwd.R)["w"][-1, 1]
        wb = run(make_model(m, **over), bwd.X, bwd.R)["w"][-1, 1]
        out["order_test"].append({"model": label, "wB_forward": float(wf),
                                  "wB_backward": float(wb)})
    # (d) Diagnostic: WHY is BB non-monotonic in tau2? For each tau2, record the
    # A-B correlation entering phase 2, the cross-gain on the first B+ trial,
    # and w_A after 1, 5 and 30 B+ trials.
    out["bb_diagnostic"] = []
    test, _ = sch.backward_blocking(N_TRIALS)
    for t2 in [0.0, 0.001, 0.01, 0.03, 0.1, 1.0]:
        h = run(make_model("Kalman", tau2=t2), test.X, test.R)
        P = h["S"][N_TRIALS - 1] + t2 * np.eye(3)      # covariance entering B+ trial 1
        out["bb_diagnostic"].append({
            "tau2": t2, "corr_AB": float(P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])),
            "k_A_first": float(h["k"][N_TRIALS, 0]),
            "wA_1": float(h["w"][N_TRIALS, 0]), "wA_5": float(h["w"][N_TRIALS + 4, 0]),
            "wA_30": float(h["w"][-1, 0])})
    return out


# ---------------------------------------------------------------------------
# Exp 3: robustness to reward noise
# ---------------------------------------------------------------------------
def exp_robustness():
    out = {}
    for m in MODEL_NAMES:
        out[m] = {}
        for p in sch.PAIRED:
            effs = np.array([paired_effect(m, p, noise_seed=s)[0] for s in range(N_SEEDS)])
            out[m][p] = {"mean": float(effs.mean()), "sd": float(effs.std(ddof=1)),
                         "n_positive": int((effs > 0).sum()), "values": effs.tolist()}
    return out


# ---------------------------------------------------------------------------
def run_all(save_dir="results"):
    results = {
        "params": {"N_TRIALS": N_TRIALS, "RW": RW_PARAMS, "Kalman": KF_PARAMS,
                   "RW+within": WC_PARAMS, "N_SEEDS": N_SEEDS,
                   "REWARD_NOISE_SD": REWARD_NOISE_SD},
        "replication": exp_replication(),
        "dk_replication": exp_dk_replication(),
        "sweep": exp_noise_sweep(),
        "ablation": exp_ablation(),
        "robustness": exp_robustness(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f)
    write_summary(results, os.path.join(save_dir, "summary.md"))
    return results


def write_summary(res, path):
    """Human-readable tables of the key numbers (copy these into the report)."""
    rep, rob = res["replication"], res["robustness"]
    # Round tiny floating-point leftovers (e.g. -1e-17) to a clean 0.
    for m in MODEL_NAMES:
        for p in rep["effects"][m]:
            if abs(rep["effects"][m][p]) < 1e-12:
                rep["effects"][m][p] = 0.0
    lines = ["# Summary of results", "",
             f"Default parameters: {json.dumps(res['params'])}", "",
             "## Exp 0 -- effect sizes, deterministic rewards "
             "(positive = phenomenon present)", "",
             "| Model | blocking | overshadowing | backward blocking | unovershadowing |"
             " acq. trials to 90% | ext. trials to 50% | w_A after extinction |",
             "|---|---|---|---|---|---|---|---|"]
    for m in MODEL_NAMES:
        e = rep["effects"][m]
        lines.append(f"| {m} | {e['blocking']:.3f} | {e['overshadowing']:.3f} | "
                     f"{e['backward_blocking']:.3f} | {e['unovershadowing']:.3f} | "
                     f"{rep['acq_trials'][m]:.0f} | {rep['ext_trials'][m]:.0f} | "
                     f"{rep['ext_final'][m]:.3f} |")
    lines += ["", f"## Exp 3 -- robustness: mean +/- SD over {N_SEEDS} seeds, "
              f"reward noise SD {REWARD_NOISE_SD} (n seeds with effect > 0)", "",
              "| Model | blocking | overshadowing | backward blocking | unovershadowing |",
              "|---|---|---|---|---|"]
    for m in MODEL_NAMES:
        cells = [f"{rob[m][p]['mean']:.3f} +/- {rob[m][p]['sd']:.3f} "
                 f"({rob[m][p]['n_positive']}/{N_SEEDS})" for p in sch.PAIRED]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    sw = res["sweep"]
    bb = np.array(sw["bb_full"])
    lines += ["", "## Exp 1 -- sweep ranges (Kalman, full covariance)", "",
              f"- BB effect (30 B+ trials): min {bb.min():.3f}, max {bb.max():.3f}",
              f"- BB effect (5 B+ trials): min {np.min(sw['bb_early']):.3f}, "
              f"max {np.max(sw['bb_early']):.3f}",
              f"- BB at tau2 = 0 (30 B+ trials): min {bb[:, 0].min():.3f}, "
              f"max {bb[:, 0].max():.3f}",
              f"- UO effect: min {np.min(sw['uo_full']):.3f}, max {np.max(sw['uo_full']):.3f}",
              f"- FB effect: min {np.min(sw['fb_full']):.3f}, max {np.max(sw['fb_full']):.3f}",
              f"- Kalman-diag BB: max |effect| {np.abs(sw['bb_diag']).max():.2e}",
              f"- Kalman-diag UO: max |effect| {np.abs(sw['uo_diag']).max():.2e}",
              f"- Kalman-diag FB: min {np.min(sw['fb_diag']):.3f}, max {np.max(sw['fb_diag']):.3f}",
              f"- Acquisition trials to 90%: min {np.nanmin(sw['acq']):.0f}, "
              f"max {np.nanmax(sw['acq']):.0f}"]
    dk = res["dk_replication"]["Kalman"]
    lines += ["", "## Exp 0b -- Dayan & Kakade (2001) Fig 2C-D replication "
              f"(params {DK_PARAMS})", "",
              f"- w after 20 AB+: A {dk['w'][19][0]:.3f}, B {dk['w'][19][1]:.3f}",
              f"- w after 20 A+: A {dk['w'][-1][0]:.3f}, B {dk['w'][-1][1]:.3f}",
              f"- S_AA at trial 20 {dk['S_AA'][19]:.3f}, at end {dk['S_AA'][-1]:.3f}",
              f"- S_AB at trial 20 {dk['S_AB'][19]:.3f}, at end {dk['S_AB'][-1]:.4f}",
              f"- Kalman-diag w_B at end: "
              f"{res['dk_replication']['Kalman-diag']['w'][-1][1]:.3f}"]
    lines += ["", "## Exp 2d -- why BB is non-monotonic in tau2 (sigma2 = "
              f"{KF_PARAMS['sigma2']})", "",
              "| tau2 | corr(A,B) entering B+ | k_A on 1st B+ trial | w_A after 1 B+ |"
              " after 5 | after 30 |", "|---|---|---|---|---|---|"]
    for d in res["ablation"]["bb_diagnostic"]:
        lines.append(f"| {d['tau2']} | {d['corr_AB']:+.3f} | {d['k_A_first']:+.3f} | "
                     f"{d['wA_1']:.3f} | {d['wA_5']:.3f} | {d['wA_30']:.3f} |")
    lines += ["", "## Exp 2e -- trial order: same trials, forward (A+ -> AB+) vs "
              "backward (AB+ -> A+) order; final w_B (lower = more blocking)", "",
              "| Model | w_B forward order | w_B backward order |", "|---|---|---|"]
    for d in res["ablation"]["order_test"]:
        lines.append(f"| {d['model']} | {d['wB_forward']:.3f} | {d['wB_backward']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_all()
    with open(os.path.join("results", "summary.md")) as f:
        print(f.read())
