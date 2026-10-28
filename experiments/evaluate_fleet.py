"""Evaluate a policy in the fleet environment (most likely action for every boat), optionally save a GIF.

Works for fleet runs and for solo runs (the solo network has the same observation layout, so a solo
policy can be dropped into the fleet unchanged: the baseline before fleet training).

Usage:
    uv run python experiments/evaluate_fleet.py runs/train/<run> [--fleet-config CFG] [--gif PATH]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from sail_env.train.analysis import mean_ci
from sail_env.config import EnvParams
from sail_env.fleet_env import FleetStartEnv, rollout_fleet
from sail_env.render import render_fleet_gif
from sail_env.agents.policy import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--fleet-config", type=Path, default=Path("experiments/configs/fleet_v1.yaml"),
                    help="fleet env settings to evaluate in (default: fleet_v1)")
    ap.add_argument("--episodes", type=int, default=1024, help="per seed")
    ap.add_argument("--gif", type=Path)
    args = ap.parse_args()

    _, _, actor, params, seeds = load_policy(args.run)
    raw = yaml.safe_load(args.fleet_config.read_text(encoding="utf-8"))
    p = EnvParams(**raw["env_params"]).replace(random_start_time=False)  # standard: everybody at T-90
    env = FleetStartEnv(p)
    greedy = lambda ap_: (lambda k, obs: jnp.argmax(actor.apply(ap_, obs), -1))  # noqa: E731
    run = jax.jit(jax.vmap(lambda ap_, k: rollout_fleet(env, p, k, greedy(ap_)), in_axes=(None, 0)))

    keys_metrics = ["good_start_on_line", "success_strict", "started", "ocs", "unserved_foul", "clear_air"]
    rows = {k: [] for k in keys_metrics + ["fouls_per_boat", "contacts_per_boat", "mark_hits_per_boat",
                                           "spread_along_line_m", "below_line_at_gun_m"]}
    print(f"{args.run.name} in a {env.num_agents}-boat fleet ({args.episodes} starts per seed, standard T{p.t_start:.0f})")
    for seed in range(seeds):
        ap_ = jax.tree.map(lambda x: x[seed], params["actor"])
        traj, acts, rewards, info = run(ap_, jax.random.split(jax.random.PRNGKey(500 + seed), args.episodes))
        info = jax.tree.map(np.asarray, info)
        for k in keys_metrics:
            rows[k].append(float(info[k].mean()))
        rows["fouls_per_boat"].append(float(info["fouls"].mean()))
        rows["contacts_per_boat"].append(float(info["contacts"].mean()))
        rows["mark_hits_per_boat"].append(float(info["mark_hits"].mean()))
        # At the gun (frame where the clock reads 0): how spread out along the line, how far below it.
        gun = int(np.argmin(np.abs(np.asarray(traj.boats.t[0, :, 0]))))
        b = jax.tree.map(lambda x: np.asarray(x[:, gun]), traj.boats.boat)
        ln = jax.tree.map(lambda x: np.asarray(x[:, gun]), traj.boats.line)
        ldx, ldy = ln.cb_x - ln.pin_x, ln.cb_y - ln.pin_y
        llen = np.hypot(ldx, ldy)
        along = ((b.x - ln.pin_x) * ldx + (b.y - ln.pin_y) * ldy) / llen
        below = -((b.x - ln.pin_x) * (-ldy) + (b.y - ln.pin_y) * ldx) / llen
        rows["spread_along_line_m"].append(float(along.std(axis=1).mean()))
        rows["below_line_at_gun_m"].append(float(np.median(below)))
        print(f"  seed {seed}: " + "  ".join(f"{k} {rows[k][-1]:.1%}" for k in keys_metrics)
              + f"  penalties/boat {rows['fouls_per_boat'][-1]:.2f}  contacts/boat {rows['contacts_per_boat'][-1]:.2f}"
              + f"  mark hits/boat {rows['mark_hits_per_boat'][-1]:.3f}")
        if args.gif and seed == 0:
            i = int(np.argmax(info["good_start_on_line"].sum(axis=1)))  # the episode with most good starts
            t1 = jax.tree.map(lambda x: x[i], traj)
            render_fleet_gif(t1, p, args.gif, title=f"{args.run.name}, seed 0: "
                             f"{int(info['good_start_on_line'][i].sum())}/{env.num_agents} good starts")
            print(f"  wrote {args.gif}")
    print("  mean +- 95% CI over seeds: " + "  ".join(
        f"{k} {mean_ci(np.array(v)[:, None])[0][0]:.3g} +- {mean_ci(np.array(v)[:, None])[1][0]:.2g}" for k, v in rows.items()))


if __name__ == "__main__":
    main()
