"""Compare training runs: success and OCS rate (mean and 95% CI over seeds) on shared axes.

Each run is given by its name (as shown in W&B / the results folder). Metrics come from the local
results folder when metrics.npz exists, otherwise from W&B (per-seed keys "seed<i>/<metric>").

Usage:
    uv run python experiments/compare_runs.py --out docs/figures/phase2_reward_comparison.png \\
        solo_sparse_20261017-132600 solo_shaped_20261017-131648 ...
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sail_env.train.analysis import ffill, mean_ci, smooth  # noqa: E402

# Runs trained with `python -m sail_env.train.run` are written to runs/train/; older runs (sailstart
# package, tag sailstart-final) to experiments/results/. W&B: new runs log to project "sail_env".
RESULTS = [Path("runs") / "train", Path(__file__).parent / "results"]
PROJECTS = ["sail_env", "sailstart"]
METRICS = ["success_rate", "ocs_rate"]
SUCCESS_KEY = None  # set by --success-key: which logged metric to plot as "success"


def load_local(name):
    f = next((d / name / "metrics.npz" for d in RESULTS if (d / name / "metrics.npz").exists()), None)
    if f is None:
        return None
    m = np.load(f)
    data = {k: m[k] for k in METRICS}
    if SUCCESS_KEY and SUCCESS_KEY in m:
        data["success_rate"] = m[SUCCESS_KEY]
    elif "success_strict_rate" in m:  # curriculum runs: compare on success under the full rules
        data["success_rate"] = m["success_strict_rate"]
    return m["steps"], data


def load_wandb(name):
    import wandb

    api = wandb.Api()
    runs = [r for project in PROJECTS for r in api.runs(f"{api.default_entity}/{project}") if r.name == name]
    if not runs:
        raise SystemExit(f"run {name!r} not found locally or in W&B")
    run = runs[0]
    n_seeds = run.config["ppo"]["num_seeds"]
    source = {k: k for k in METRICS}
    if SUCCESS_KEY and f"seed0/{SUCCESS_KEY}" in run.summary:
        source["success_rate"] = SUCCESS_KEY
    elif "seed0/success_strict_rate" in run.summary:  # curriculum runs: success under the full rules
        source["success_rate"] = "success_strict_rate"
    keys = ["env_steps"] + [f"seed{i}/{source[k]}" for k in METRICS for i in range(n_seeds)]
    rows = sorted(run.scan_history(keys=keys), key=lambda r: r["env_steps"])
    steps = np.array([r["env_steps"] for r in rows])
    data = {k: np.array([[r[f"seed{i}/{source[k]}"] for r in rows] for i in range(n_seeds)]) for k in METRICS}
    return steps, data


def label_of(name):
    return name.rsplit("_", 1)[0]  # drop the timestamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/comparison.png"))
    ap.add_argument("--window", type=int, default=10, help="smoothing window in updates")
    ap.add_argument("--success-key", help="logged metric to plot as success, e.g. good_start_on_line_rate")
    args = ap.parse_args()
    global SUCCESS_KEY
    SUCCESS_KEY = args.success_key

    loaded = {}
    for name in args.runs:
        loaded[name] = load_local(name) or load_wandb(name)
        print(f"{name}: {loaded[name][1]['success_rate'].shape[1]} updates "
              f"({'local' if load_local(name) else 'W&B'})")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.5 * len(METRICS), 3.8))
    print(f"\n{'run':28s} {'final success':>22s} {'final OCS':>22s}   (last 10% of updates, mean +- 95% CI)")
    for name, (steps, data) in loaded.items():
        row = []
        for ax, k in zip(axes, METRICS):
            x = ffill(data[k])
            m, h = mean_ci(smooth(x, args.window))
            ax.plot(steps, m, lw=1.5, label=label_of(name))
            ax.fill_between(steps, m - h, m + h, alpha=0.2)
            tail = x[:, -max(len(steps) // 10, 1):].mean(axis=1)  # per-seed final value
            fm, fh = mean_ci(tail[:, None])
            row.append(f"{100 * fm[0]:7.2f}% +- {100 * fh[0]:5.2f}%")
        print(f"{label_of(name):28s} {row[0]:>22s} {row[1]:>22s}")
    for ax, k in zip(axes, METRICS):
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.set_xlabel("env steps per seed")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    n = next(iter(loaded.values()))[1]["success_rate"].shape[0]
    fig.suptitle(f"Solo start: {' vs '.join(label_of(r) for r in loaded)} (mean, 95% CI over {n} seeds)", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
