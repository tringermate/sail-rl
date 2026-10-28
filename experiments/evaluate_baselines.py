"""Digital baselines vs the RL boats, on the same standard starts.

Solo: the digital sailor (fixed settings) and the digital expert (best of 40 settings per start, tried
in the simulator) in the environment of a trained solo run, next to that run's RL policy.
Fleet: five digital sailors aiming at spots spread along the line, in a fleet config, next to the RL fleet.

Usage:
    uv run python experiments/evaluate_baselines.py --solo-run runs/train/<solo run>
    uv run python experiments/evaluate_baselines.py --fleet-config experiments/configs/fleet_c4.yaml [--gif PATH]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from sail_env.agents.digital_sailor import digital_expert, digital_sailor_action, rollout_solo
from sail_env.config import EnvParams
from sail_env.fleet_env import FleetStartEnv, rollout_fleet
from sail_env.render import render_fleet_gif
from sail_env.start_env import SoloStartEnv
from sail_env.agents.policy import load_policy

METRICS = ["good_start_on_line", "success_strict", "started", "ocs"]


def report(name, info):
    info = jax.tree.map(np.asarray, info)
    line = "  ".join(f"{k} {info[k].mean():6.1%}" for k in METRICS if k in info)
    extra = ""
    if "upwind_efficiency" in info:
        ok = info["success_strict"].astype(bool)
        extra = f"  upwind eff. of successes {info['upwind_efficiency'][ok].mean():.2f}" if ok.any() else ""
    print(f"  {name:28s} {line}{extra}")


def solo(run: Path, episodes: int):
    _, p, _, _, _ = load_policy(run)
    p = p.replace(random_start_time=False)
    env = SoloStartEnv(p)
    keys = jax.random.split(jax.random.PRNGKey(7), episodes)
    _, states = jax.vmap(env.reset, in_axes=(0, None))(keys, p)
    print(f"Solo, {episodes} standard T{p.t_start:.0f} starts, env of {run.name}:")
    sailor, _ = jax.jit(jax.vmap(lambda s: rollout_solo(env, p, s, 0.3, -3.0)))(states)
    report("digital sailor", sailor)
    expert, _, best = jax.jit(jax.vmap(lambda s: digital_expert(env, p, s)))(states)
    report("digital expert (best of 40)", expert)
    print("  (RL: see experiments/evaluate.py --greedy on the same run)")


def fleet(cfg_path: Path, episodes: int, gif):
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    p = EnvParams(**raw["env_params"]).replace(random_start_time=False)
    env = FleetStartEnv(p)
    n = env.num_agents
    spots = jnp.linspace(0.15, 0.85, n)  # each digital sailor aims at its own spot on the line

    def policy(k, obs, fs):
        return jax.vmap(lambda s, spot: digital_sailor_action(s, env.polar, p, spot, -3.0))(fs.boats, spots)

    # rollout_fleet hands the policy only observations, so step with the state directly here.
    def run_one(key):
        k_reset, k_run = jax.random.split(key)
        obs, fs = env.reset(k_reset, p)

        def body(carry, k):
            obs, fs, finished = carry
            acts = policy(k, obs, fs)
            new_obs, new_fs, r, done, info = env.step_env(k, fs, acts, p)
            new_fs = jax.tree.map(lambda a, b: jnp.where(finished, b, a), new_fs, fs)
            return (new_obs, new_fs, finished | done[0]), (new_fs, info, done[0] & ~finished)

        (_, _, _), (traj, infos, ends) = jax.lax.scan(body, (obs, fs, jnp.bool_(False)), jax.random.split(k_run, p.n_steps))
        traj = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), fs, traj)
        last = jnp.argmax(ends)
        return traj, jax.tree.map(lambda x: x[last], infos)

    traj, info = jax.jit(jax.vmap(run_one))(jax.random.split(jax.random.PRNGKey(11), episodes))
    print(f"Fleet of {n} digital sailors, {episodes} standard starts, env of {cfg_path.name}:")
    report("digital sailors", info)
    inf = jax.tree.map(np.asarray, info)
    print(f"  contacts/boat {inf['contacts'].mean():.2f}  mark hits/boat {inf['mark_hits'].mean():.3f}  "
          f"penalties/boat {inf['fouls'].mean():.2f}  clear air {inf['clear_air'].mean():.1%}")
    if gif:
        i = int(np.argmax(inf["good_start_on_line"].sum(axis=1)))
        render_fleet_gif(jax.tree.map(lambda x: x[i], traj), p, gif,
                         title=f"digital sailors: {int(inf['good_start_on_line'][i].sum())}/{n} good starts")
        print(f"  wrote {gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-run", type=Path)
    ap.add_argument("--fleet-config", type=Path)
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--gif", type=Path)
    args = ap.parse_args()
    if args.solo_run:
        solo(args.solo_run, args.episodes)
    if args.fleet_config:
        fleet(args.fleet_config, min(args.episodes, 512), args.gif)


if __name__ == "__main__":
    main()
