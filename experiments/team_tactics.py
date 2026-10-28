"""Do team tactics appear? Race an agent against itself in 2v2 team races and measure what team racers do
differently from individual racers (compare F1, trained for its own place, with T1, trained for the team).

    python experiments/team_tactics.py F1=runs/train/<f1> T1=runs/train/<t1> --races 256 --out runs/team_tactics \
        [--env race-weather] [--set rrs16=true current_max_frac=0.06]

All four boats are the same agent; the races are scored as team races (boats 0-1 vs 2-3) and are the same
races for every agent. Per agent:

    winning combinations   how the winning team won (2v2: 1-2, 1-3, 2-3; 2-3 beats 1-4 on the tie-break).
                           Team racers win more by 2-3: holding the opponents' second boat back behind both.
    shadow on opponents    seconds per race an opponent sails in this team's wind shadow, vs the same for a
                           teammate: covering / slowing opponents instead of just sailing fast. The ratio is per
                           boat (an opponent's seconds / 2 opponents vs the teammate's): 1 = no preference
    pass-backs             races where the team's leader at the first mark finishes behind its teammate
    leader's time          finish time of the race winner: a team racer may sail slower to manage the race
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sail_env as se  # noqa: E402
from sail_env.agents import load_agent, pretrained_agent  # noqa: E402
from sail_env.fleet_rules import shadow_factors  # noqa: E402

COMBOS = {(1, 2): "1-2", (1, 3): "1-3", (2, 3): "2-3", (1, 4): "1-4", (2, 4): "2-4", (3, 4): "3-4"}


def make_agent(spec):
    path, _, seed = spec.partition("#")
    return pretrained_agent(path, seed=int(seed)) if seed else load_agent(path)


def races(env, p, policy, keys):
    n = env.num_agents
    team = jnp.arange(n) // p.team_size

    def one(key):
        obs, s = env.reset(key, p)

        def body(c, i):
            o, s, fin, shade, last = c
            a = jax.vmap(policy)(jax.random.split(jax.random.fold_in(key, i), n), o, s.boats, jnp.arange(n))
            wd, tws = env.local_winds(s.boats, p)
            racing = (s.boats.t >= 0.0) & (s.leg < 4)
            in_shadow = (shadow_factors(s.boats.boat, wd, p, tws) < 1.0) & racing[:, None]  # [i, j]: i in j's shadow
            o2, s2, _, d, info = env.step_env(jax.random.PRNGKey(0), s, a, p)
            end = d[0] & ~fin
            last = jax.tree.map(lambda new, old: jnp.where(end, new, old), info, last)
            shade = shade + jnp.where(fin, 0.0, in_shadow.astype(jnp.float32))
            s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)
            return (o2, s2, fin | d[0], shade, last), None

        _, _, _, _, info0 = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(n, jnp.int32), p)
        init = (obs, s, jnp.bool_(False), jnp.zeros((n, n)), jax.tree.map(jnp.zeros_like, info0))
        (_, s_end, _, shade, last), _ = jax.lax.scan(body, init, jnp.arange(p.n_steps))
        same = team[:, None] == team[None, :]
        off = ~jnp.eye(n, dtype=bool)
        return {"place": last["race_rank"] + 1.0, "mark1": last["mark1_place"], "finish": s_end.finish_time,
                "team_win": last["team_win"],
                "shadow_from_opp": jnp.sum(jnp.where(~same, shade, 0.0), axis=1),  # s boat i spent in an opponent's shadow
                "shadow_from_mate": jnp.sum(jnp.where(same & off, shade, 0.0), axis=1)}

    return jax.tree.map(np.asarray, jax.jit(jax.vmap(one))(keys))


def ci95(x):
    """Mean and normal-approximation 95% interval of per-race values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def summarise(out, team_size):
    place, mark1, win = out["place"], out["mark1"], out["team_win"]
    r = len(place)
    combos, passbacks, two_three = {}, np.zeros(r), np.zeros(r)
    for k in range(r):
        w = 0 if win[k, 0] else 1
        mine = place[k, w * team_size:(w + 1) * team_size]
        c = COMBOS.get(tuple(sorted(int(round(x)) for x in mine)), "other")
        combos[c] = combos.get(c, 0) + 1
        two_three[k] = c == "2-3"
        for t in (0, 1):
            m = mark1[k, t * team_size:(t + 1) * team_size]
            f = place[k, t * team_size:(t + 1) * team_size]
            if np.all(m > 0) and m[0] != m[1] and np.argmin(m) != np.argmin(f):
                passbacks[k] += 1
    winner = np.array([np.min(np.where(np.isfinite(f), f, np.nan)) for f in out["finish"]])
    # per boat: seconds in one opponent's shadow (2 opponents) over seconds in the teammate's; per race, then averaged
    opp = out["shadow_from_opp"].sum(1) / (team_size * out["shadow_from_opp"].shape[1])
    mate = out["shadow_from_mate"].sum(1) / out["shadow_from_mate"].shape[1]
    ratio = float(opp.mean() / max(mate.mean(), 1e-6))
    boot = np.random.default_rng(0).integers(0, r, (2000, r))
    ratio_ci = np.percentile(opp[boot].mean(1) / np.maximum(mate[boot].mean(1), 1e-6), [2.5, 97.5]).tolist()
    return {"races": r, "winning_combinations": {k: v / r for k, v in sorted(combos.items())},
            "two_three_ci95": ci95(two_three),
            "pass_backs_per_race": float(passbacks.mean()),  # per race, both teams counted
            "pass_backs_ci95": ci95(passbacks),
            "shadow_from_opponents_s": float(out["shadow_from_opp"].mean()),
            "shadow_from_teammate_s": float(out["shadow_from_mate"].mean()),
            "shadow_ratio_per_boat": ratio, "shadow_ratio_ci95": ratio_ci,
            "winner_time_s": float(np.nanmean(winner)), "winner_time_ci95": ci95(winner),
            "finished": float(np.isfinite(out["finish"]).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+", help="name=spec (a run directory, #seed optional, or a built-in agent)")
    ap.add_argument("--races", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/team_tactics"))
    ap.add_argument("--env", default="race")
    ap.add_argument("--set", nargs="*", default=["rrs16=true"], metavar="KEY=VALUE", help="env_params overrides")
    args = ap.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    env, p = se.make(args.env, fleet_size=4, team_size=2, **overrides)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.races)
    results = {}
    for name, spec in (a.split("=", 1) for a in args.agents):
        results[name] = summarise(races(env, p, make_agent(spec)(env, p), keys), p.team_size)
        print(name, json.dumps(results[name]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "team_tactics.json").write_text(json.dumps({"env": args.env, "overrides": overrides,
                                                            "races": args.races, "results": results}, indent=1),
                                                encoding="utf-8")

    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["1-2", "1-3", "2-3"]
    w = 0.8 / len(names)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#333333"]
    for i, n in enumerate(names):
        c = results[n]["winning_combinations"]
        ax[0].bar(np.arange(3) + i * w, [c.get(k, 0.0) for k in labels], w, label=n, color=colors[i % 5])
        ax[1].bar(np.arange(2) + i * w, [results[n]["shadow_from_opponents_s"], results[n]["shadow_from_teammate_s"]], w,
                  color=colors[i % 5])
        ax[2].bar(i, results[n]["pass_backs_per_race"], color=colors[i % 5])
    ax[0].set_xticks(np.arange(3) + w * (len(names) - 1) / 2, labels)
    ax[0].set_title("How the winning team won (share of races)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].set_xticks(np.arange(2) + w * (len(names) - 1) / 2, ["in an opponent's shadow", "in the teammate's shadow"])
    ax[1].set_title("Seconds per boat per race", fontsize=10)
    ax[2].set_xticks(range(len(names)), names)
    ax[2].set_title("Pass-backs per race (leader at mark 1 finishes behind its teammate)", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e3e3e3", lw=0.6)
    sets = " ".join(f"{k}={v}" for k, v in overrides.items())
    fig.suptitle(f"Team tactics: each agent sails all 4 boats, 2v2 scoring, {args.races} races ({args.env}, {sets})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "team_tactics.png", dpi=110)
    print(f"wrote {args.out}/team_tactics.json and team_tactics.png")


if __name__ == "__main__":
    main()
