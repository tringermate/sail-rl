"""Failure map: where does a trained policy succeed or fail, and how?

Sails the final policies of a run from many random starts (the training distribution with random
start times, so time to gun can be mapped too), records each start's initial conditions and sorts
its outcome into one category:

    good start    on time (T+0..3), >= 90% polar speed, on starboard
    port start    on time and fast, but on port
    slow          on time, but below 90% speed
    late          legal start after T+3
    OCS           still over the line at the end
    no start      never crossed legally (stayed short, or missed the ends)

Writes a figure with the outcome breakdown and the good-start rate by initial conditions.

Usage: uv run python experiments/failure_map.py runs/train/<run> [--out PNG] [--episodes 8192]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sail_env.train.ppo import categorical_sample  # noqa: E402
from sail_env.config import KNOT  # noqa: E402
from sail_env.geometry import signed_twa  # noqa: E402
from sail_env.polar import polar_speed  # noqa: E402
from sail_env.rules import line_frame, signed_distance  # noqa: E402
from sail_env.start_env import SoloStartEnv  # noqa: E402
from sail_env.agents.policy import load_policy  # noqa: E402

OUTCOMES = ["good start", "port start", "slow", "late", "OCS", "no start"]
COLORS = ["#2a9d55", "#8fbf3f", "#e9c46a", "#f4a261", "#e76f51", "#9aa0a6"]


def classify(info, p):
    """Outcome category per episode (index into OUTCOMES), from the info of its final step."""
    on_time = info["started"] & (info["start_time"] <= p.strict_success_window)
    fast = info["start_speed_ratio"] >= p.strict_success_speed_frac
    cat = np.full(info["started"].shape, 5)  # no start
    cat[info["ocs"]] = 4
    cat[info["started"] & ~on_time] = 3
    cat[on_time & ~fast] = 2
    cat[on_time & fast & ~info["start_on_starboard"]] = 1
    cat[on_time & fast & info["start_on_starboard"]] = 0
    return cat


def rollout_final(env, p, actor, actor_params, key, n):
    """Run n episodes to their end (lengths differ); return the initial states and each episode's final info."""
    _, s0 = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, n), p)
    step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))

    def body(carry, k):
        s, finished, final = carry
        obs = jax.vmap(env.get_obs, in_axes=(0, None))(s, p)
        a = categorical_sample(k, actor.apply(actor_params, obs))
        _, s2, _, done, info = step(jax.random.split(k, n), s, a, p)
        first_end = done & ~finished
        final = jax.tree.map(lambda new, old: jnp.where(first_end, new, old), info, final)
        keep = finished
        s2 = jax.tree.map(lambda new, old: jnp.where(keep.reshape((-1,) + (1,) * (new.ndim - 1)), old, new), s2, s)
        return (s2, finished | done, final), None

    _, _, _, _, info0 = step(jax.random.split(key, n), s0, jnp.zeros(n, jnp.int32), p)  # shape template
    init = (s0, jnp.zeros(n, bool), jax.tree.map(jnp.zeros_like, info0))
    (_, _, final), _ = jax.lax.scan(body, init, jax.random.split(key, p.n_steps))
    return s0, final


def initial_conditions(env, s0):
    below = -jax.vmap(signed_distance)(s0.line, s0.boat.x, s0.boat.y)
    dx, dy, _, _, _ = jax.vmap(line_frame)(s0.line)
    centre_x, centre_y = (s0.line.pin_x + s0.line.cb_x) / 2, (s0.line.pin_y + s0.line.cb_y) / 2
    lateral = (s0.boat.x - centre_x) * dx + (s0.boat.y - centre_y) * dy  # > 0 towards the committee boat
    twa = jax.vmap(signed_twa)(s0.boat.heading, s0.wind.base_dir)
    v_polar = jax.vmap(lambda a, tws: polar_speed(env.polar, a, tws))(twa, s0.wind.tws)
    return {
        "time_to_gun": -s0.t,
        "below": below,
        "tws_kn": s0.wind.tws / KNOT,
        "lateral": lateral,
        "abs_twa_deg": jnp.degrees(jnp.abs(twa)),
        "speed_frac": s0.boat.speed / jnp.maximum(v_polar, 1e-3),
    }


def rate_heatmap(ax, x, y, good, xbins, ybins, xlabel, ylabel, min_count=30):
    hx, hy = np.digitize(x, xbins) - 1, np.digitize(y, ybins) - 1
    rate = np.full((len(ybins) - 1, len(xbins) - 1), np.nan)
    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            m = (hx == i) & (hy == j)
            if m.sum() >= min_count:
                rate[j, i] = good[m].mean()
    # Draw cells on an index grid (bins are uneven), labelled with the bin edges.
    im = ax.imshow(rate, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    for i in range(len(xbins) - 1):
        for j in range(len(ybins) - 1):
            if not np.isnan(rate[j, i]):
                ax.text(i, j, f"{rate[j, i]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if rate[j, i] < 0.6 else "black")
    ax.set_xticks(np.arange(len(xbins)) - 0.5, [f"{b:g}" for b in np.round(xbins, 0)], fontsize=7)
    ax.set_yticks(np.arange(len(ybins)) - 0.5, [f"{b:g}" for b in np.round(ybins, 0)], fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    return im


def rate_curve(ax, x, good, bins, xlabel):
    idx = np.digitize(x, bins) - 1
    centres = (bins[:-1] + bins[1:]) / 2
    rate = [good[idx == i].mean() if (idx == i).sum() >= 30 else np.nan for i in range(len(bins) - 1)]
    ax.plot(centres, rate, "o-")
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("good-start rate", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=8192, help="per seed")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cfg, p, actor, params, seeds = load_policy(args.run)
    p = p.replace(random_start_time=True)  # map time to gun as well
    env = SoloStartEnv(p)
    run_fn = jax.jit(lambda ap_, k: rollout_final(env, p, actor, ap_, k, args.episodes))

    conds, cats = [], []
    for seed in range(seeds):
        s0, final = run_fn(jax.tree.map(lambda x: x[seed], params["actor"]), jax.random.PRNGKey(2000 + seed))
        conds.append(jax.tree.map(np.asarray, initial_conditions(env, s0)))
        cats.append(classify(jax.tree.map(np.asarray, final), p))
    c = {k: np.concatenate([d[k] for d in conds]) for k in conds[0]}
    cat = np.concatenate(cats)
    good = cat == 0

    print(f"{args.run.name}: {len(cat)} starts ({seeds} seeds x {args.episodes}), random start times")
    for i, name in enumerate(OUTCOMES):
        print(f"  {name:10s} {np.mean(cat == i):6.1%}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    shares = [np.mean(cat == i) for i in range(len(OUTCOMES))]
    ax.barh(OUTCOMES[::-1], shares[::-1], color=COLORS[::-1])
    for i, v in enumerate(shares[::-1]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title("outcome of all starts", fontsize=9)
    dist_bins = np.array([0, 10, 20, 40, 60, 80, 100, 120.01])
    im = rate_heatmap(axes[0, 1], c["time_to_gun"], c["below"], good, np.array([5, 10, 20, 30, 40, 50, 60.01]),
                      dist_bins, "time to gun at start (s)", "distance below line at start (m)")
    axes[0, 1].set_title("good-start rate: time to gun x distance", fontsize=9)
    rate_heatmap(axes[0, 2], c["tws_kn"], c["below"], good, np.array([6, 8, 10, 12, 14, 16.01]), dist_bins,
                 "wind speed (kn)", "distance below line at start (m)")
    axes[0, 2].set_title("good-start rate: wind x distance", fontsize=9)
    rate_curve(axes[1, 0], c["lateral"], good, np.linspace(-100, 100, 11), "sideways offset at start (m, + = committee end)")
    rate_curve(axes[1, 1], c["abs_twa_deg"], good, np.array([0, 30, 45, 60, 90, 120, 150, 180.01]),
               "initial |TWA| (deg; < 30 = in irons)")
    rate_curve(axes[1, 2], c["speed_frac"], good, np.linspace(0, 1.0001, 6), "initial speed / polar")
    fig.colorbar(im, ax=axes[0, 1:], shrink=0.8, label="good-start rate")
    fig.suptitle(f"Failure map: {args.run.name} ({len(cat)} starts, all seeds, random start times)", fontsize=10)
    out = args.out or Path("docs/figures") / f"failure_map_{args.run.name.rsplit('_', 1)[0]}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
