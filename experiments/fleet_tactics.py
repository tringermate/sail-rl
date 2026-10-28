"""Is there racing, or a procession? Fleet-tactics measures for agents racing themselves (all boats the same agent).

    python experiments/fleet_tactics.py digital=digital F2=runs/train/<run> --races 128 --out runs/fleet_tactics
        [--teams] [--set rrs16=true current_max_frac=0.06] [--env race-weather]

Every race is the same for every agent (same seeds). Per agent, averaged over races:

    split_m           mean spread across the course (std of the boats' cross-course positions) while beating:
                      ~0 = everyone on the same line (follow-the-leader), large = boats taking different sides
    overtakes         pairwise order swaps per race after the first windward mark (by distance still to sail)
    cover_tacks       per race: the leader tacks within 15 s after a boat within 150 m behind it tacked
    shadow_s          seconds per boat per race in another boat's wind shadow
    finish_s, finished
and a GIF of race 0 (runs/fleet_tactics/<agent>_race0.gif).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import sail_env as se  # noqa: E402
from sail_env.agents import load_agent, pretrained_agent  # noqa: E402
from sail_env.geometry import signed_twa  # noqa: E402
from sail_env.race_env import distance_to_go  # noqa: E402

COVER_WINDOW, COVER_RANGE = 15, 150.0


def make_agent(spec):
    path, _, seed = spec.partition("#")
    return pretrained_agent(path, seed=int(seed)) if seed else load_agent(path)


def record_races(env, p, policy, keys):
    n = env.num_agents

    def one(key):
        obs, s = env.reset(key, p)
        k = env.course(s, p)

        def body(c, i):
            o, s, fin = c
            a = jax.vmap(policy)(jax.random.split(jax.random.fold_in(key, i), n), o, s.boats, jnp.arange(n))
            b = s.boats.boat
            wd, _ = env.local_winds(s.boats, p)
            racing = (s.boats.t >= 0.0) & (s.leg < 4) & ~fin
            rec = {"cross": (b.x - k["c"][0]) * k["r"][0] + (b.y - k["c"][1]) * k["r"][1],
                   "togo": s.leg * 1e4 - distance_to_go(s.leg, b.x, b.y, k, 4, s.gate_through),  # higher = ahead
                   "leg": s.leg, "tack": signed_twa(b.heading, wd) >= 0.0, "racing": racing}
            o2, s2, _, d, _ = env.step_env(jax.random.PRNGKey(0), s, a, p)
            s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)
            return (o2, s2, fin | d[0]), rec

        (_, s_end, _), rec = jax.lax.scan(body, (obs, s, jnp.bool_(False)), jnp.arange(p.n_steps))
        return rec, s_end.finish_time, s_end.shadow_time

    rec, ft, shadow = jax.jit(jax.vmap(one))(keys)
    return jax.tree.map(np.asarray, rec), np.asarray(ft), np.asarray(shadow)


def measures(rec, ft, shadow):
    races, steps, n = rec["leg"].shape
    split, overtakes, covers = [], [], []
    for r in range(races):
        racing, leg, cross, togo, tack = (rec[k][r] for k in ("racing", "leg", "cross", "togo", "tack"))
        beat = racing & ((leg == 0) | (leg == 2))
        both = beat.sum(axis=1) >= 2
        split.append(np.mean([np.std(cross[t][beat[t]]) for t in np.where(both)[0]]) if both.any() else np.nan)
        # pairwise order swaps after every boat has rounded the first mark, while all are racing
        live = np.where(racing.all(axis=1) & (leg >= 1).all(axis=1))[0]
        swaps = 0
        for a in range(n):
            for b in range(a + 1, n):
                ahead = togo[live, a] > togo[live, b]
                swaps += int(np.count_nonzero(np.diff(ahead.astype(int))))
        overtakes.append(swaps)
        # cover tacks: the leader tacks within COVER_WINDOW s after a boat behind it (within COVER_RANGE) tacked
        tacked = np.zeros_like(tack)
        tacked[1:] = (tack[1:] != tack[:-1]) & beat[1:] & beat[:-1]
        c = 0
        for t in np.where(tacked.any(axis=1))[0]:
            if not beat[t].any():
                continue
            lead = int(np.argmax(np.where(racing[t], togo[t], -np.inf)))
            for j in np.where(tacked[t])[0]:
                if j == lead or abs(togo[t, lead] - togo[t, j]) > COVER_RANGE:
                    continue
                if tacked[t + 1:t + 1 + COVER_WINDOW, lead].any():
                    c += 1
        covers.append(c)
    fin = np.isfinite(ft)
    return {"split_m": float(np.nanmean(split)), "overtakes": float(np.mean(overtakes)),
            "cover_tacks": float(np.mean(covers)), "shadow_s": float(shadow.mean()),
            "finish_s": float(np.mean(ft[fin])) if fin.any() else None, "finished": float(fin.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--env", default="race")
    ap.add_argument("--boats", type=int, default=4)
    ap.add_argument("--teams", action="store_true")
    ap.add_argument("--set", nargs="*", default=["rrs16=true"])
    ap.add_argument("--races", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("runs/fleet_tactics"))
    args = ap.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    if args.teams:
        overrides["team_size"] = args.boats // 2
    env, p = se.make(args.env, fleet_size=args.boats, **overrides)
    keys = jax.random.split(jax.random.PRNGKey(0), args.races)
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in (a.split("=", 1) for a in args.agents):
        policy = make_agent(spec)(env, p)
        results[name] = measures(*record_races(env, p, policy, keys))
        print(name, json.dumps(results[name]))
        from sail_env.runs import params_dict, render_replay, run_episode

        record, _ = jax.jit(lambda k: run_episode(env, p, k, [policy], jnp.zeros(env.num_agents, jnp.int32)))(keys[0])
        render_replay({"record": jax.tree.map(np.asarray, record), "params": params_dict(p), "name": name},
                      args.out / f"{name}_race0.gif", title=f"{name}: all {args.boats} boats, race 0")
    (args.out / "fleet_tactics.json").write_text(json.dumps({"env": args.env, "overrides": overrides,
                                                             "races": args.races, "results": results}, indent=1))
    names = list(results)
    keys_ = ["split_m", "overtakes", "cover_tacks", "shadow_s"]
    titles = ["spread across the course on the beats (m)", "overtakes per race", "cover tacks per race",
              "s in another boat's shadow"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, k, t in zip(ax, keys_, titles):
        a.bar(names, [results[n][k] for n in names], color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#333333"][:len(names)])
        a.set_title(t, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Fleet tactics, {args.boats} boats sailed by the same agent, {args.races} races ({args.env}, "
                 f"{' '.join(args.set)}{' teams' if args.teams else ''})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "fleet_tactics.png", dpi=110)
    print(f"wrote {args.out}/fleet_tactics.json, fleet_tactics.png and the race-0 GIFs")


if __name__ == "__main__":
    main()
