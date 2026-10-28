"""Phase 3 figures: success curves of the samplers, and the sampling weight of six tracked
starting situations (trivial -> too hard) over training, as in the SGS paper.

Usage: uv run python experiments/plot_phase3.py
"""

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from sail_env.train.analysis import ffill, mean_ci, smooth  # noqa: E402

RUNS = ["p3_uniform_512", "p3_sfl_512", "p3_sgs_512", "p3_sgs66_512", "p3_uniform_4k", "p3_sfl_4k", "p3_sgs_4k"]
FOLLOW_UP = ["p3_staged_uniform", "p3_staged_sgs", "p3_sgs_sharp", "p3_sfl_512"]


def load(name):
    d = sorted(glob.glob(f"runs/train/{name}_2026*") + glob.glob(f"experiments/results/{name}_2026*"))[-1]
    return np.load(f"{d}/metrics.npz"), yaml.safe_load(open(f"{d}/config.yaml", encoding="utf-8"))


fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, suffix in zip(axes, ["_512", "_4k"]):
    for name in [r for r in RUNS if r.endswith(suffix)]:
        m, _ = load(name)
        mean, half = mean_ci(smooth(ffill(m["good_start_on_line_rate"]), 10))
        ax.plot(m["steps"], mean, label=name.replace("p3_", "").replace(suffix, ""))
        ax.fill_between(m["steps"], mean - half, mean + half, alpha=0.2)
    ax.set_title(f"{suffix[1:]} envs: good starts on the sampled situations", fontsize=9)
    ax.set_xlabel("env steps per seed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Phase 3: uniform vs SFL vs SGS (mean, 95% CI over 5 seeds)", fontsize=10)
fig.tight_layout()
fig.savefig("docs/figures/phase3_success.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, name in zip(axes, ["p3_sfl_512", "p3_sgs_512", "p3_sgs66_512"]):
    m, cfg = load(name)
    reach = cfg.get("tracked_reach_fraction", [None] * 6)
    for k in range(6):
        w = smooth(ffill(m[f"tracked{k}_weight"]), 10).mean(axis=0)
        ax.plot(m["steps"], w, label=f"reach {reach[k]:.2f}" if reach[k] is not None else f"config {k}")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_title(f"{name}: sampling weight (1 = uniform share)", fontsize=9)
    ax.set_xlabel("env steps per seed")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, title="tracked situation\n(reach > 1: impossible)", title_fontsize=7)
fig.tight_layout()
fig.savefig("docs/figures/phase3_weights.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in FOLLOW_UP:
    m, _ = load(name)
    key = "good_start_on_line_rate"
    mean, half = mean_ci(smooth(ffill(m[key]), 10))
    axes[0].plot(m["steps"], mean, label=name.replace("p3_", ""))
    axes[0].fill_between(m["steps"], mean - half, mean + half, alpha=0.2)
axes[0].set_title("good starts (strict) on the sampled situations", fontsize=9)
axes[0].set_xlabel("env steps per seed"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
m, cfg = load("p3_staged_sgs")
reach = cfg.get("tracked_reach_fraction")
for k in range(6):
    axes[1].plot(m["steps"], smooth(ffill(m[f"tracked{k}_weight"]), 10).mean(axis=0), label=f"reach {reach[k]:.2f}")
axes[1].axhline(1.0, color="gray", lw=0.8, ls="--"); axes[1].set_yscale("log")
axes[1].set_title("staged_sgs: sampling weight of tracked situations (1 = uniform)", fontsize=9)
axes[1].set_xlabel("env steps per seed"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("docs/figures/phase3_follow_up.png", dpi=120)
print("wrote docs/figures/phase3_success.png, phase3_weights.png, phase3_follow_up.png")
