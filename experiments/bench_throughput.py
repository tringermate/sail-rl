"""How fast does training run here? Environment steps per second for a training config at several numbers
of parallel environments (CPU vs GPU, before moving training to a GPU pod).

    python experiments/bench_throughput.py experiments/configs/ladder/s4k_full_kick.yaml --envs 64 512 2048 4096

For each num_envs it builds the run's training (same env, start bank, teacher, PPO settings; only num_envs
changed, one seed), compiles it, then times --chunks chunks of updates. Compile time is reported apart.
Steps per second = env steps (all envs, all seeds) per wall-clock second, the number that sets how long a
run takes: a 40M-step seed at S steps/s takes 40e6 / S seconds.
Results are appended to runs/bench_throughput.csv (device, config, num_envs, steps/s, compile s).
"""

import argparse
import csv
import time
from pathlib import Path

import jax
import yaml

from sail_env.train.curriculum.config_env import ConfigResetEnv, Curriculum, make_race_start_bank
from sail_env.train.curriculum.sampler import SamplerConfig
from sail_env.train.ppo import PPOConfig, make_train
from sail_env.train.run import make_env, make_teacher
from sail_env.wrappers import LogWrapper


def build(raw, num_envs, num_seeds):
    ppo = dict(raw.get("ppo", {}), num_envs=num_envs, num_seeds=num_seeds, total_timesteps=10**9)
    cfg = PPOConfig(**ppo)
    env, env_params = make_env(raw["env"], raw.get("env_params", {}))
    raw_env, curriculum = env, None
    if "start_bank" in raw:
        sb = raw["start_bank"]
        bank = make_race_start_bank(raw_env, env_params, jax.random.PRNGKey(sb.get("seed", 4321)), sb.get("num", 4096),
                                    sb.get("policy", "fleet"))
        curriculum = Curriculum(SamplerConfig(method=sb.get("sampler", "uniform"), num_configs=sb.get("num", 4096)),
                                success_key="race_finished")
        env = ConfigResetEnv(raw_env, bank)
    env = LogWrapper(env)
    teacher = make_teacher(raw["teacher"], raw_env, env_params) if raw.get("teacher") and cfg.kick_coef > 0 else None
    init_fn, chunk_fn = make_train(cfg, env, env_params, None, curriculum, teacher)
    return cfg, init_fn, chunk_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--envs", type=int, nargs="+", default=[64, 512, 2048, 4096])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=3, help="timed chunks (of ppo.updates_per_chunk updates) after compiling")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = str(jax.devices()[0])
    print(f"device: {device} ({jax.default_backend()}), config {args.config.name}")
    out = Path("runs") / "bench_throughput.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    for n in args.envs:
        cfg, init_fn, chunk_fn = build(raw, n, args.seeds)
        keys = jax.random.split(jax.random.PRNGKey(0), args.seeds)
        t0 = time.perf_counter()
        runner = jax.jit(jax.vmap(init_fn))(keys)
        chunk = jax.jit(jax.vmap(chunk_fn))
        runner, m = chunk(runner)  # compile + first chunk
        jax.block_until_ready(m)
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(args.chunks):
            runner, m = chunk(runner)
        jax.block_until_ready(m)
        dt = time.perf_counter() - t0
        steps = args.chunks * cfg.updates_per_chunk * cfg.batch_size * args.seeds
        sps = steps / dt
        print(f"num_envs {n:5d}: {sps:10,.0f} steps/s  (compile + first chunk {compile_s:5.1f} s; "
              f"3 seeds x 40M steps at this rate: {3 * 40e6 / sps / 3600:5.2f} h)", flush=True)
        with out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["device", "config", "num_envs", "seeds", "steps_per_s", "compile_s"])
                new = False
            w.writerow([device, args.config.name, n, args.seeds, round(sps), round(compile_s, 1)])


if __name__ == "__main__":
    main()
