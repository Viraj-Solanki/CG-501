"""
make_figures.py -- ONE command regenerates every result and figure:

    python make_figures.py

It runs all experiments (experiments.run_all), writes results/results.json
and results/summary.md, and saves figures/fig1..fig6 as PNG (300 dpi).
Everything is deterministic (fixed seeds), so reruns give identical output.
"""

import os

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

import experiments as ex
import schedules as sch

FIG_DIR = "figures"

# One fixed colour per model, used in every figure (colour follows the model).
COLORS = {"RW": "#2a78d6", "Kalman": "#eb6834", "Kalman-diag": "#1baf7a",
          "RW+within": "#52514e"}
# Kalman-diag is dashed and drawn on top of the (thicker) Kalman line, so
# both stay visible where they coincide. Line style = secondary encoding.
STYLES = {"RW": "-", "Kalman": "-", "Kalman-diag": "--", "RW+within": "-."}
WIDTHS = {"RW": 1.6, "Kalman": 2.8, "Kalman-diag": 1.6, "RW+within": 1.6}
LABELS = {"RW": "Rescorla-Wagner", "Kalman": "Kalman (full covariance)",
          "Kalman-diag": "Kalman (diagonal / ablated)",
          "RW+within": "RW + within-compound"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]   # light -> dark, one hue

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def shade_phases(ax, phases):
    """Vertical lines + labels at phase boundaries (from a Schedule)."""
    for label, start, end in phases:
        if start > 0:
            ax.axvline(start + 0.5, color=INK2, lw=0.8, ls=":")
        ax.text((start + end) / 2 + 0.5, 1.01, label, ha="center", va="bottom",
                fontsize=8, color=INK2, transform=ax.get_xaxis_transform())


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", os.path.join(FIG_DIR, name))


# ---------------------------------------------------------------------------
def fig1_replication(res):
    """Critical-cue weight over trials, 6 paradigms, RW vs Kalman vs ablation."""
    rep = res["replication"]["trajectories"]
    N = ex.N_TRIALS
    panels = [  # (paradigm key, title, cue index, schedule for phase labels)
        ("acquisition", "Acquisition: $w_A$", 0, sch.acquisition(N)),
        ("extinction", "Extinction: $w_A$", 0, sch.extinction(N)),
        ("blocking", "Forward blocking: $w_B$", 1, sch.blocking(N)[0]),
        ("overshadowing", "Overshadowing: $w_A$", 0, sch.overshadowing(N)[0]),
        ("backward_blocking", "Backward blocking: $w_A$", 0, sch.backward_blocking(N)[0]),
        ("unovershadowing", "Unovershadowing: $w_B$", 1, sch.unovershadowing(N)[0]),
    ]
    models = ["RW", "Kalman", "Kalman-diag"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
    for ax, (key, title, cue, s) in zip(axes.flat, panels):
        for m in models:
            w = np.array(rep[m][key]["test"])[:, cue]
            t = np.arange(1, len(w) + 1)
            ax.plot(t, w, STYLES[m], color=COLORS[m], lw=WIDTHS[m], label=LABELS[m])
            if "control" in rep[m][key]:
                wc = np.array(rep[m][key]["control"])[:, cue]
                ax.plot(np.arange(1, len(wc) + 1), wc, ":", color=COLORS[m], lw=1.0)
        shade_phases(ax, s.phases)
        ax.set_title(title, pad=20)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("trial")
    for ax in axes[:, 0]:
        ax.set_ylabel("associative weight")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls=":", color=INK2, lw=1.0))
    labels.append("control condition (dotted)")
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig1_replication.png")


def fig2_effects(res):
    """Effect sizes (test - control) for each paradigm and model, noisy rewards."""
    rob = res["robustness"]
    paradigms = list(sch.PAIRED)
    names = ["Forward\nblocking", "Overshadowing", "Backward\nblocking", "Unovershadowing"]
    models = ex.MODEL_NAMES
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(paradigms))
    for i, m in enumerate(models):
        means = [rob[m][p]["mean"] for p in paradigms]
        sds = [rob[m][p]["sd"] for p in paradigms]
        ax.bar(x + (i - 1.5) * width, means, width * 0.92, yerr=sds,
               color=COLORS[m], label=LABELS[m], capsize=2,
               error_kw=dict(ecolor=INK2, lw=0.8),
               hatch="///" if m == "RW+within" else None, edgecolor="white", lw=0)
        for xi, mu in zip(x + (i - 1.5) * width, means):
            if abs(mu) < 1e-9:                       # make "exactly zero" visible
                ax.text(xi, 0.012, "0", ha="center", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("effect size (test - control)")
    ax.set_title(f"Effect sizes: mean $\\pm$ SD over {ex.N_SEEDS} seeds "
                 f"(reward noise SD = {ex.REWARD_NOISE_SD})")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 0.75)
    save(fig, "fig2_effect_sizes.png")


def fig3_mechanism(res):
    """Inside the Kalman filter during backward blocking (AB+ -> B+)."""
    rep = res["replication"]
    w = np.array(rep["trajectories"]["Kalman"]["backward_blocking"]["test"])
    S = np.array(rep["bb_kalman_S"])
    k = np.array(rep["bb_kalman_k"])
    s = sch.backward_blocking(ex.N_TRIALS)[0]
    t = np.arange(1, len(w) + 1)
    cA, cB = "#eb6834", "#2a78d6"   # cue A / cue B (cue identity, not model)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    axes[0].plot(t, w[:, 0], color=cA, lw=2.8, label="$w_A$ (absent in phase 2)")
    axes[0].plot(t, w[:, 1], color=cB, label="$w_B$")
    axes[0].set_title("(a) Mean weights", pad=20)
    axes[0].set_ylabel("weight")

    axes[1].plot(t, S[:, 0, 0], color=cA, lw=2.8, label="$\\Sigma_{AA}$ (var A)")
    axes[1].plot(t, S[:, 1, 1], color=cB, label="$\\Sigma_{BB}$ (var B)")
    axes[1].plot(t, S[:, 0, 1], color=INK, ls="--", label="$\\Sigma_{AB}$ (covariance)")
    axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set_title("(b) Posterior (co)variance", pad=20)

    axes[2].plot(t, k[:, 0], color=cA, lw=2.8, label="$k_A$")
    axes[2].plot(t, k[:, 1], color=cB, label="$k_B$")
    axes[2].axhline(0, color=INK2, lw=0.6)
    axes[2].set_title("(c) Kalman gain (learning rate)", pad=20)

    for ax in axes:
        shade_phases(ax, s.phases)
        ax.set_xlabel("trial")
        ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save(fig, "fig3_kalman_mechanism.png")


def _heatmap(ax, data, tau2, sigma2, title, cbar_label, fmt, fig, vmin=None, vmax=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd"] + BLUES[1:] + ["#0d366b"])
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(tau2)), ["0"] + [f"{v:.0e}".replace("e-0", "e-").replace("e+00", "")
                                             for v in tau2[1:]], rotation=60, fontsize=7)
    ax.set_yticks(range(len(sigma2)), [f"{v:.2g}" for v in sigma2], fontsize=7)
    ax.set_xlabel("process noise $\\tau^2$")
    ax.set_ylabel("observation noise $\\sigma^2$")
    ax.set_title(title)
    ax.grid(False)
    lo, hi = np.nanmin(data), np.nanmax(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            frac = (v - lo) / (hi - lo + 1e-12)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5,
                    color="white" if frac > 0.6 else INK)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)


def fig4_sweep(res):
    sw = res["sweep"]
    tau2, sigma2 = np.array(sw["tau2"]), np.array(sw["sigma2"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _heatmap(axes[0], np.array(sw["bb_full"]), tau2, sigma2,
             "(a) Backward blocking, 30 B+ trials", "effect on $w_A$", "{:.2f}", fig, 0, 0.5)
    _heatmap(axes[1], np.array(sw["bb_early"]), tau2, sigma2,
             "(b) Backward blocking, 5 B+ trials", "effect on $w_A$", "{:.2f}", fig, 0, 0.5)
    _heatmap(axes[2], np.array(sw["acq"]), tau2, sigma2,
             "(c) Acquisition: trials to reach $w_A$ = 0.9", "trials", "{:.0f}", fig)
    fig.tight_layout()
    save(fig, "fig4_noise_sweep.png")


def fig5_ablation(res):
    ab = res["ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # (a) BB vs tau2, full vs diagonal. symlog x-axis so tau2 = 0 is visible.
    t = np.array(ab["tau2_fine"])
    for m in ["Kalman", "Kalman-diag"]:
        axes[0].plot(t, ab[f"bb_vs_tau2_{m}"], "-o", ms=3, color=COLORS[m], label=LABELS[m])
    axes[0].axhline(0, color=COLORS["RW"], lw=1.0, ls="--")
    axes[0].text(2e-4, 0.03, "RW and ablated Kalman: exactly 0", color=INK2, fontsize=7.5)
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set_xlim(0, 1.2)
    axes[0].set_xlabel("process noise $\\tau^2$  ($\\sigma^2$ = %.1f)" % ex.KF_PARAMS["sigma2"])
    axes[0].set_ylabel("backward-blocking effect")
    axes[0].set_ylim(-0.05, 0.55)
    axes[0].set_title("(a) Ablation removes backward blocking")
    axes[0].legend(fontsize=7.5, loc="center left")

    # (b) BB vs length of AB+ phase, several tau2 (one-hue ramp: light = small tau2)
    N1 = np.array(ab["N1_values"])
    for c, (t2, effs) in zip(BLUES, ab["bb_vs_N1"].items()):
        axes[1].plot(N1, effs, color=c, label=f"$\\tau^2$ = {t2}")
    axes[1].set_xlabel("number of AB+ trials (phase 1)")
    axes[1].set_ylabel("backward-blocking effect")
    axes[1].set_title("(b) Effect vs. compound training")
    axes[1].legend(fontsize=7.5)

    # (c) Covariance S_AB during AB+ training.
    for c, (t2, cov) in zip(BLUES, ab["cov_AB_vs_N1"].items()):
        axes[2].plot(np.arange(1, len(cov) + 1), cov, color=c, label=f"$\\tau^2$ = {t2}")
    axes[2].set_xlabel("number of AB+ trials")
    axes[2].set_ylabel("$\\Sigma_{AB}$ after AB+ training")
    axes[2].set_title("(c) Cross-cue covariance")
    axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_ablation.png")


def fig6_alternative(res):
    """Kalman vs. the non-Bayesian alternative (RW + within-compound)."""
    rep = res["replication"]["trajectories"]
    N = ex.N_TRIALS
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), sharey=True)
    for ax, (key, cue, title, s) in zip(axes, [
            ("backward_blocking", 0, "Backward blocking: $w_A$", sch.backward_blocking(N)[0]),
            ("unovershadowing", 1, "Unovershadowing: $w_B$", sch.unovershadowing(N)[0])]):
        for m in ["RW", "Kalman", "RW+within"]:
            w = np.array(rep[m][key]["test"])[:, cue]
            ax.plot(np.arange(1, len(w) + 1), w, STYLES[m], color=COLORS[m], lw=WIDTHS[m],
                    label=LABELS[m])
        shade_phases(ax, s.phases)
        ax.set_title(title, pad=20)
        ax.set_xlabel("trial")
    axes[0].set_ylabel("associative weight")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig6_alternative_model.png")


def fig7_dk_replication(res):
    """Replication of Dayan & Kakade (2001), Fig 2C-D (exact Kalman filter)."""
    dk = res["dk_replication"]
    s = sch.make_schedule("dk2001", [("AB", 1, 20), ("A", 1, 20)])
    t = np.arange(1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    w = np.array(dk["Kalman"]["w"])
    axes[0].plot(t, w[:, 0], color=INK, label="$w_A$ (their light L)")
    axes[0].plot(t, w[:, 1], color=INK, ls="--", label="$w_B$ (their sound S)")
    wd = np.array(dk["Kalman-diag"]["w"])
    axes[0].plot(t, wd[:, 1], color=COLORS["Kalman-diag"], ls="--",
                 label="$w_B$, diagonal ablation")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("weight")
    axes[0].set_title("(a) Mean weights", pad=20)
    axes[1].plot(t, dk["Kalman"]["S_AA"], color=INK, label="$\\Sigma_{AA}$")
    axes[1].plot(t, dk["Kalman"]["S_AB"], color=INK, ls="--", label="$\\Sigma_{AB}$")
    axes[1].axhline(0, color=INK2, lw=0.6)
    axes[1].set_ylim(-0.1, 0.1)
    axes[1].set_title("(b) Posterior covariance", pad=20)
    for ax in axes:
        shade_phases(ax, s.phases)
        ax.set_xlabel("trial")
        ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    save(fig, "fig7_dk2001_replication.png")


if __name__ == "__main__":
    results = ex.run_all()
    fig1_replication(results)
    fig2_effects(results)
    fig3_mechanism(results)
    fig4_sweep(results)
    fig5_ablation(results)
    fig6_alternative(results)
    fig7_dk_replication(results)
    print("\nAll figures in ./figures, numbers in ./results/summary.md")
