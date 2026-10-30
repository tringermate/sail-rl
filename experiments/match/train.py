"""Train a 1v1 (match racing) policy on the situation library, from a YAML config.

    python -m experiments.match.train experiments/configs/match/m1_uniform.yaml [--no-wandb] [--total-timesteps N]

Like sail_env.train.run (same PPO, logger, outputs), with the environment of experiments/match/env.py:
config keys `env` (a race preset; fleet_size must be 2), `env_params` (EnvParams overrides only, so
sail_env's load_policy can read the run), `bank` (situation bank), `match` (MatchParams), `curriculum`
(sampler), `anneal` ({field: [start, end, from, until]}, a MatchParams field or an EnvParams field), `ppo`,
`init_from`. Writes runs/train/<name>_<timestamp>/ with config.yaml, metrics.npz, params.msgpack, curves.png.
"""

import argparse
import dataclasses
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import serialization

from experiments.match.env import MatchCurriculum, MatchEnv, MatchParams
from experiments.match.situations import NAMES, build_bank
from sail_env.presets import make
from sail_env.train.analysis import plot_curves
from sail_env.train.curriculum.sampler import SamplerConfig
from sail_env.train.logger import Logger
from sail_env.train.ppo import PPOConfig, train
from sail_env.train.run import widen_input_layer
from sail_env.wrappers import LogWrapper

RESULTS = Path("runs") / "train"
MATCH_FIELDS = {f.name for f in dataclasses.fields(MatchParams)} - {"env", "cdf"}


def setup(raw, log=print):
    """(raw race env, MatchEnv, MatchParams, curriculum, n_sit) for a config dict."""
    race_env, env_params = make(raw.get("env", "race"), **raw.get("env_params", {}))
    b = raw.get("bank", {})
    t = time.perf_counter()
    log("building the situation bank (the digital sailor sails every setup alone, for the undisturbed times):")
    bank, n_sit = build_bank(race_env, env_params, jax.random.PRNGKey(b.get("seed", 2024)), b.get("per_situation", 512),
                             b.get("start_checkpoint", 50.0), b.get("full_races", 1024), b.get("situations", NAMES),
                             start_policy=b.get("start_policy", "fleet"), log=log)
    log(f"bank: {n_sit} situation entries (setups x 3 role pairings) + {int(bank.kind.shape[0]) - n_sit} full races "
        f"in {time.perf_counter() - t:.0f} s")
    env = MatchEnv(race_env, bank, n_sit, b.get("start_checkpoint", 50.0))
    params = MatchParams(env=env_params, cdf=jnp.linspace(1.0 / n_sit, 1.0, n_sit), **raw.get("match", {}))
    sc = SamplerConfig(**{**raw.get("curriculum", {}), "num_configs": n_sit})
    return race_env, env, params, MatchCurriculum(sc, bank, n_sit), n_sit


def make_anneal(params: MatchParams, anneal: dict):
    """env_params_fn for PPO: fields move linearly from start to end between the training fractions from/until."""
    def fn(progress):
        values, env_values = {}, {}
        for name, (start, end, a, b) in anneal.items():
            x = jnp.clip((progress - a) / max(b - a, 1e-9), 0.0, 1.0)
            (values if name in MATCH_FIELDS else env_values)[name] = start + (end - start) * x
        p = params.replace(env=params.env.replace(**env_values), **values)
        return p, {**values, **env_values}

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, help="override ppo.total_timesteps (e.g. for a quick test)")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.total_timesteps:
        raw.setdefault("ppo", {})["total_timesteps"] = args.total_timesteps
    unknown = set(raw.get("match", {})) - MATCH_FIELDS
    if unknown:
        raise ValueError(f"unknown match parameters {sorted(unknown)}; choose from {sorted(MATCH_FIELDS)}")
    cfg = PPOConfig(**raw.get("ppo", {}))
    race_env, env, params, curriculum, n_sit = setup(raw)
    env = LogWrapper(env)

    out = RESULTS / f"{raw['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    resolved = {f.name: getattr(params.env, f.name) for f in dataclasses.fields(params.env)}
    resolved = {k: (float(v) if hasattr(v, "dtype") else v) for k, v in resolved.items()}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    full_config = {**raw, "ppo": dataclasses.asdict(cfg), "seed": args.seed, "env_params_resolved": resolved,
                   "match_resolved": {k: float(getattr(params, k)) for k in sorted(MATCH_FIELDS)}, "git_commit": commit}
    (out / "config.yaml").write_text(yaml.safe_dump(full_config, sort_keys=False), encoding="utf-8")
    print(f"{raw['name']}: {cfg.num_seeds} seeds x {cfg.num_updates} updates x {cfg.batch_size} steps "
          f"= {cfg.num_updates * cfg.batch_size:,} steps per seed on {jax.devices()[0]}  ->  {out}", flush=True)
    logger = Logger(out.name, full_config, raw.get("wandb", False) and not args.no_wandb,
                    project=raw.get("wandb_project", "sail_env"))

    init_params = None
    if raw.get("init_from"):  # e.g. the bundled racer: new inputs (the match features) start with weight 0
        from sail_env.agents.policy import load_policy

        src = Path(raw["init_from"])
        if not src.exists():
            src = Path(__import__("sail_env").__file__).parent / "pretrained" / raw["init_from"]
        _, _, _, src_params, src_seeds = load_policy(src)
        init_params = jax.tree.map(lambda x: x[jnp.arange(cfg.num_seeds) % src_seeds], src_params)
        n_in = init_params["actor"]["params"]["Dense_0"]["kernel"].shape[1]
        if n_in < env.obs_dim:
            init_params, at = widen_input_layer(init_params, env.obs_dim, params.env)
            print(f"initialised from {src} ({src_seeds} seeds); input layer widened {n_in} -> {env.obs_dim}")

    checkpoint_every = raw.get("checkpoint_every", 100)

    def callback(update, runner):
        if checkpoint_every and (update + 1) % checkpoint_every == 0:
            (out / "params.msgpack").write_bytes(serialization.to_bytes(jax.device_get(runner.params)))
            (out / "checkpoint.txt").write_text(f"update {update + 1}, {(update + 1) * cfg.batch_size:,} steps per seed\n",
                                                encoding="utf-8")

    t0 = time.perf_counter()
    runner, metrics = train(cfg, env, params, jax.random.PRNGKey(args.seed), log_fn=logger.log, chunk_fn_callback=callback,
                            env_params_fn=make_anneal(params, raw.get("anneal", {})), curriculum=curriculum,
                            init_params=init_params)
    elapsed = time.perf_counter() - t0
    steps = (np.arange(cfg.num_updates) + 1) * cfg.batch_size
    np.savez(out / "metrics.npz", steps=steps, **{k.replace("/", "__"): v for k, v in metrics.items()})
    (out / "params.msgpack").write_bytes(serialization.to_bytes(jax.device_get(runner.params)))
    keys = ["episode_return", "episode_length", "block_success", "contacts_per_boat", "entropy", "explained_variance"]
    plot_curves(metrics, steps, [k for k in keys if k in metrics], out / "curves.png", title=raw["name"])
    print(f"done in {elapsed:.0f}s. block success (last update, per seed): {np.round(metrics['block_success'][:, -1], 2)}",
          flush=True)
    try:
        logger.finish()
    except KeyboardInterrupt:
        print("warning: W&B shutdown was interrupted; results were already saved locally", flush=True)


if __name__ == "__main__":
    main()
