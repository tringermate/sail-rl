"""Evaluate a match policy: every situation and role on a held-out bank, and full 1v1 races, against an opponent.

    python -m experiments.match.evaluate runs/train/m1_uniform_<stamp> --vs runs/train/m0_control_<stamp>
    python -m experiments.match.evaluate runs/train/m1_uniform_<stamp> --vs digital --gifs 1
    python -m experiments.match.evaluate digital --vs digital --config experiments/configs/match/m1_uniform.yaml

A and B are a run directory (optionally '#seed'), 'digital' (the hand-written racer: it always races, it never
blocks), or for --vs also 'self'. The held-out bank uses the run's bank settings with another seed (bank seed +
1). Each setup is sailed twice, A as boat 0 and as boat 1. Reports per situation: A's win rate when both race,
A's block success (A blocks, B escapes) and escape success (A escapes, B blocks), contacts and penalties per
episode; and A's win rate in full 1v1 races (from just after the start). Policies act greedily.
Writes runs/match_eval/<A>_vs_<B>.md and .json (and GIFs with --gifs).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.match.env import MATCH_FEATURES
from experiments.match.situations import BLOCK, ESCAPE, FULL_RACE, NAMES, PAIRINGS, RACE
from experiments.match.train import make_anneal, setup
from sail_env.agents import digital_sailor_agent
from sail_env.agents.policy import load_policy

OUT = Path("runs") / "match_eval"


def make_actor(spec, race_env, env_params):
    """act(key, obs (2, D_match), fs) -> (2,) actions for both boats (the caller picks which boat is whose)."""
    if spec == "digital":
        pol = digital_sailor_agent(race_env, env_params)
        d = race_env.obs_dim
        return lambda key, obs, fs: jax.vmap(pol, in_axes=(None, 0, 0, 0))(key, obs[:, :d], fs.boats, jnp.arange(2))
    path, _, seed = spec.partition("#")
    _, _, actor, params, seeds = load_policy(Path(path))
    p = jax.tree.map(lambda x: x[int(seed or 0) % seeds], params["actor"])
    n_in = p["params"]["Dense_0"]["kernel"].shape[0]
    if n_in not in (race_env.obs_dim, race_env.obs_dim + MATCH_FEATURES):
        raise ValueError(f"{spec}: {n_in} inputs, expected a race policy ({race_env.obs_dim}) or a match policy "
                         f"({race_env.obs_dim + MATCH_FEATURES})")
    return lambda key, obs, fs: jnp.argmax(actor.apply(p, obs[:, :n_in]), axis=-1)


def play(env, params, act_a, act_b, idx, a_boat, steps):
    """Sail bank entries idx (vmapped), A as boat a_boat. Returns the final info of each episode."""
    def one(i, ab):
        obs, s = env.entry(i, jnp.int32(0), params)

        def body(carry, k):
            obs, s, over, last = carry
            a = jnp.where(jnp.arange(2) == ab, act_a(k, obs, s.fs), act_b(k, obs, s.fs))
            obs2, s2, _, done, info = env.step_env(k, s, a, params)
            keep = lambda new, old: jax.tree.map(lambda x, y: jnp.where(over, y, x), new, old)  # noqa: E731
            return (keep(obs2, obs), keep(s2, s), over | done[0], keep(info, last)), None

        *_, info0 = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(2, jnp.int32), params)
        info0 = jax.tree.map(jnp.zeros_like, info0)
        (_, _, over, info), _ = jax.lax.scan(body, (obs, s, jnp.bool_(False), info0),
                                            jax.random.split(jax.random.PRNGKey(1), steps))
        return info, over

    return jax.jit(jax.vmap(one, in_axes=(0, None)))(jnp.asarray(idx), a_boat)


def render(env, params, race_env, act_a, act_b, i, path, title, steps):
    """One episode of bank entry i (A = boat 0) as a GIF."""
    from sail_env.runs import params_dict, render_replay, snapshot

    @jax.jit
    def run():
        obs, s = env.entry(i, jnp.int32(0), params)

        def body(carry, k):
            obs, s, over = carry
            a = jnp.where(jnp.arange(2) == 0, act_a(k, obs, s.fs), act_b(k, obs, s.fs))
            obs2, s2, _, done, _ = env.step_env(k, s, a, params)
            s2 = jax.tree.map(lambda x, y: jnp.where(over, y, x), s2, s)
            return (obs2, s2, over | done[0]), (snapshot(race_env, params.env, s2.fs), over)

        _, (snaps, over) = jax.lax.scan(body, (obs, s, jnp.bool_(False)), jax.random.split(jax.random.PRNGKey(2), steps))
        return snapshot(race_env, params.env, s.fs), snaps, over

    first, snaps, over = run()
    n = int(np.argmax(np.asarray(over))) if np.asarray(over).any() else steps
    record = jax.tree.map(lambda a, b: np.concatenate([np.asarray(a)[None], np.asarray(b)[:n]]), first, snaps)
    render_replay({"record": record, "params": params_dict(params.env), "name": title}, path, title=title)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("--vs", default="self")
    ap.add_argument("--config", help="config for the env and bank (default: A's run config)")
    ap.add_argument("--per-situation", type=int, default=128, help="held-out setups per situation")
    ap.add_argument("--full-races", type=int, default=128)
    ap.add_argument("--gifs", type=int, default=0, help="GIFs per situation and pairing (A = boat 0)")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else Path(args.a.partition("#")[0]) / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    b = raw.setdefault("bank", {})
    b.update(seed=b.get("seed", 2024) + 1, per_situation=args.per_situation, full_races=args.full_races)
    race_env, env, params, _, n_sit = setup(raw)
    params = make_anneal(params, raw.get("anneal", {}))(1.0)[0]  # the end-of-training values (e.g. block_factor)
    vs = args.a if args.vs == "self" else args.vs
    act_a, act_b = make_actor(args.a, race_env, params.env), make_actor(vs, race_env, params.env)

    sit = np.asarray(env.bank.situation)
    role0 = np.asarray(env.bank.role[:, 0])
    rows, summary = [], {}
    for sid, name in enumerate(NAMES + ["full_race"]):
        idx = np.flatnonzero(sit == sid)
        if not len(idx):
            continue
        steps = 1300 if sid == FULL_RACE else 500
        res = {}
        for ab in (0, 1):
            info, over = play(env, params, act_a, act_b, idx, ab, steps)
            r = np.asarray(info["result"])[:, ab]
            role_a = np.asarray(env.bank.role[idx, ab])
            for key, mask in (("race", role_a == RACE), ("block", role_a == BLOCK), ("escape", role_a == ESCAPE)):
                res.setdefault(key, []).extend((r[mask] > 0).tolist())
            res.setdefault("tie", []).extend((r[role_a == RACE] == 0).tolist())
            res.setdefault("contacts", []).extend(np.asarray(info["contacts"]).sum(axis=1).tolist())
            res.setdefault("fouls", []).extend(np.asarray(info["fouls"]).sum(axis=1).tolist())
            res.setdefault("undecided", []).extend((~np.asarray(over)).tolist())
        s = {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}
        s["episodes"] = len(res["contacts"])
        summary[name] = s
        rows.append(f"| {name} | {s['race']:.2f} | {s['tie']:.2f} | {s.get('block', float('nan')):.2f} | "
                    f"{s.get('escape', float('nan')):.2f} | {s['contacts']:.2f} | {s['fouls']:.2f} | {s['episodes']} |")
        print(rows[-1], flush=True)
        if args.gifs and sid != FULL_RACE:
            OUT.mkdir(parents=True, exist_ok=True)
            for pairing in (0, 1):
                picks = idx[role0[idx] == PAIRINGS[pairing][0]][:args.gifs]
                for j, i in enumerate(picks):
                    tag = f"{name}_{'race' if pairing == 0 else 'A_blocks'}_{j}"
                    render(env, params, race_env, act_a, act_b, int(i), OUT / f"{Path(args.a).name}_vs_{Path(vs).name}_{tag}.gif",
                           f"{name}: A (boat 0) {'races' if pairing == 0 else 'blocks'}", steps)

    head = (f"# {args.a} (A) vs {vs} (B)\n\nHeld-out bank (seed {b['seed']}), each setup sailed with A as boat 0 and as "
            f"boat 1; greedy actions; block_factor {float(params.block_factor):.2f}.\n\n"
            "| situation | A wins (race) | ties | A block success | A escape success | contacts / ep | penalties / ep | episodes |\n"
            "|---|---|---|---|---|---|---|---|\n")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{Path(args.a.partition('#')[0]).name}_vs_{Path(vs.partition('#')[0]).name}"
    stem.with_suffix(".md").write_text(head + "\n".join(rows) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"-> {stem}.md")


if __name__ == "__main__":
    main()
