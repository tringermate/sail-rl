"""Evaluate a trained run's final policies on fresh starts and describe what the boats do.

By default every episode starts at T-60 (the standard task), even if the run trained with easier
starts in the mix, so runs can be compared fairly. Reports per seed and mean +- 95% CI over seeds.

Usage:
    uv run python experiments/evaluate.py runs/train/<run> [--episodes 4096] [--train-dist]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sail_env.train.ppo import categorical_sample
from sail_env.train.analysis import mean_ci
from sail_env.config import KNOT
from sail_env.geometry import signed_twa
from sail_env.physics import bow_position
from sail_env.rules import line_frame, signed_distance
from sail_env.start_env import SoloStartEnv
from sail_env.agents.policy import load_policy

ACTIONS = ["hold", "up5", "down5", "up15", "down15", "tack", "luff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--train-dist", action="store_true", help="use the run's own start distribution")
    ap.add_argument("--greedy", action="store_true", help="always take the most likely action (no sampling)")
    args = ap.parse_args()

    cfg, p, actor, params, seeds = load_policy(args.run)
    if not args.train_dist:
        p = p.replace(random_start_time=False)  # standard task: always start at T-60
    env = SoloStartEnv(p)
    n = args.episodes

    @jax.jit
    def evaluate(actor_params, key):
        _, s = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, n), p)

        def body(s, k):
            obs = jax.vmap(env.get_obs, in_axes=(0, None))(s, p)
            logits = actor.apply(actor_params, obs)
            a = jnp.argmax(logits, axis=-1) if args.greedy else categorical_sample(k, logits)
            _, s2, _, _, info = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))(jax.random.split(k, n), s, a, p)
            bx, by = jax.vmap(bow_position, in_axes=(0, None))(s2.boat, p)
            below = -jax.vmap(signed_distance)(s2.line, bx, by)
            ldx, ldy, _, _, llen = jax.vmap(line_frame)(s2.line)
            along = (bx - s2.line.pin_x) * ldx + (by - s2.line.pin_y) * ldy
            inside = (along >= 0) & (along <= llen)  # bow between the ends (along the line)
            stbd = jax.vmap(lambda b, w: signed_twa(b.heading, w.base_dir) > 0)(s2.boat, s2.wind)
            approaching = ~s2.status.started
            return s2, (s2.t, below, s2.boat.speed, a, info, stbd, approaching, inside)

        _, (t, below, speed, actions, info, stbd, appr, inside) = jax.lax.scan(body, s, jax.random.split(key, p.n_steps))
        return t, below, speed, actions, jax.tree.map(lambda x: x[-1], info), stbd, appr, inside

    rows = {k: [] for k in ["success", "good_start", "on_line_at_gun", "good_start_on_line", "started", "ocs",
                            "near_line_at_gun"]}
    print(f"{args.run.name}: {n} episodes per seed, {'training' if args.train_dist else 'standard T-60'} starts, {'greedy' if args.greedy else 'sampled'} actions")
    for seed in range(seeds):
        ap_ = jax.tree.map(lambda x: x[seed], params["actor"])
        t, below, speed, actions, info, stbd, appr, inside = jax.tree.map(np.asarray, evaluate(ap_, jax.random.PRNGKey(1000 + seed)))
        last10 = (t >= -10) & (t <= 0) & appr  # final approach, not yet started
        stbd_last10 = stbd[last10].mean() if last10.any() else float("nan")
        gun = int(np.argmin(np.abs(t[:, 0])))  # step at which the clock reads T=0 (standard starts)
        near = np.abs(below[gun]) < 10.0
        on_line = (below[gun] >= 0) & (below[gun] <= 3.0) & inside[gun]  # 0-3 m below, between the ends
        for k, v in [("success", info["success_strict"]), ("good_start", info["good_start"]),
                     ("on_line_at_gun", on_line), ("good_start_on_line", info["good_start"] & on_line),
                     ("started", info["started"]), ("ocs", info["ocs"]), ("near_line_at_gun", near)]:
            rows[k].append(float(np.mean(v)))
        hist = np.bincount(actions.ravel(), minlength=7) / actions.size
        top = ", ".join(f"{ACTIONS[i]} {hist[i]:.0%}" for i in np.argsort(-hist)[:3])
        print(f"  seed {seed}: success {rows['success'][-1]:6.1%}  started {rows['started'][-1]:6.1%}  "
              f"OCS {rows['ocs'][-1]:5.1%}  within 10 m at gun {near.mean():5.1%}  "
              f"median at gun {np.median(below[gun]):5.1f} m below, {np.median(speed[gun]) / KNOT:4.1f} kn  [{top}]")
        st = info["started"]
        if st.any():
            ok = info["success_strict"]
            eff = f"{np.mean(info['upwind_efficiency'][ok]):.2f}" if ok.any() else "-"
            print(f"          on starboard in the last 10 s {stbd_last10:5.1%}  "
                  f"starts on starboard {np.mean(info['start_on_starboard'][st]):5.1%}  "
                  f"good starts (full rules + starboard) {rows['good_start'][-1]:5.1%}  "
                  f"upwind efficiency of successes {eff}")
    print("  mean +- 95% CI over seeds: " + "  ".join(
        f"{k} {100 * mean_ci(np.array(v)[:, None])[0][0]:.1f}% +- {100 * mean_ci(np.array(v)[:, None])[1][0]:.1f}%"
        for k, v in rows.items()))


if __name__ == "__main__":
    main()
