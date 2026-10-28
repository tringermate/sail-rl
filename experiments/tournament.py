"""Tournament: trained racers (and reference agents) race each other in mixed fleets (5 boats, or --boats).

Every race draws a line-up of boats from the agent pool (each agent appears equally often over the
tournament; an agent may sail more than one boat in a race). Races use fixed seeds, so a tournament can
be rerun or extended with a new agent on exactly the same races. All races run in one jit-compiled batch.

Scores per agent (over all the boats it sailed):
    place        mean finishing place (1 = winner), 95% CI by bootstrapping over races (boats in the
                 same race are not independent)
    finished     share of races finished; finish time over the finished ones
    penalties    penalty turns per race; contacts per race
    head to head P(X beats Y) over the races where both sailed (beats = better place)

Usage:
    uv run python experiments/tournament.py A=runs/train/race_a_shaped_<time> B=runs/train/race_b_sfl_<time> \\
        digital=digital --env race-weather --races 256 --out runs/tournament/abc
    uv run python experiments/tournament.py racer=rl-racer digital=digital --env race --boats 4 --set rrs16=true
    uv run python experiments/tournament.py T1=runs/train/<t1> F1=runs/train/<f1> digital=digital --env race --boats 4 \n        --set rrs16=true --teams        # 2v2 team racing: each race is one agent's team (boats 0-1) vs another's
An agent spec is anything sail_env.agents.load_agent accepts: digital, random, hold, a run directory
(trained racers sail the start with the start policy they were trained behind), or module:function.
Append #k to a run directory to use its seed k (default 0).
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import yaml  # noqa: E402

import sail_env as se  # noqa: E402
from sail_env.agents import load_agent, pretrained_agent  # noqa: E402

BOATS = 5  # default fleet size; --boats sets it (EnvParams.fleet_size)


def make_agent(spec):
    path, _, seed = spec.partition("#")
    if seed:
        return pretrained_agent(path, seed=int(seed))
    return load_agent(path)


def line_ups(n_agents, n_races, key):
    """(n_races, BOATS) agent index per boat: a shuffled round robin, so every agent sails equally often."""
    total = n_races * BOATS
    pool = np.tile(np.arange(n_agents), -(-total // n_agents))[:total]
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    rng.shuffle(pool)
    return pool.reshape(n_races, BOATS)


def team_line_ups(n_agents, n_races, key, team_size):
    """(n_races, 2 x team_size): every ordered pair of different agents equally often, each team one agent."""
    pairs = [(a, b) for a in range(n_agents) for b in range(n_agents) if a != b]
    idx = np.tile(np.arange(len(pairs)), -(-n_races // len(pairs)))[:n_races]
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    rng.shuffle(idx)
    return np.array([[pairs[i][0]] * team_size + [pairs[i][1]] * team_size for i in idx])


def team_score(names, lineups, out, team_size):
    """Per agent: share of team races won (95% CI over races), mean team score; P(team a beats team b)."""
    win = np.asarray(out["team_win"])[:, 0].astype(float)  # team of boat 0 won
    first, second = lineups[:, 0], lineups[:, team_size]
    rows, k = {}, len(names)
    for a, name in enumerate(names):
        per_race = np.concatenate([win[first == a], 1.0 - win[second == a]])
        lo, hi = bootstrap_ci(per_race)
        rows[name] = {"team_races": int(len(per_race)), "team_win_rate": float(per_race.mean()), "ci95": [lo, hi],
                      "finished": float(np.asarray(out["race_finished"])[lineups == a].mean())}
    beats = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(k):
            m1, m2 = (first == a) & (second == b), (first == b) & (second == a)
            n = m1.sum() + m2.sum()
            if a != b and n:
                beats[a, b] = (win[m1].sum() + (1.0 - win[m2]).sum()) / n
    return rows, beats


def race_outcomes(env, params, policies, keys, lineups):
    """Final outcome of every race (no auto-reset): place, finished, finish time, legs, fouls, contacts."""
    idx = jnp.arange(env.num_agents)

    def one(key, lineup):
        k_reset, k_run = jax.random.split(key)
        obs, s = env.reset(k_reset, params)

        def body(carry, k):
            obs, s, finished, final = carry
            k_act, k_step = jax.random.split(k)
            keys_b = jax.random.split(k_act, env.num_agents)
            cand = jnp.stack([jax.vmap(p)(keys_b, obs, s.boats, idx) for p in policies])  # (P, N)
            a = cand[lineup, idx].astype(jnp.int32)
            obs2, s2, _, done, info = env.step_env(k_step, s, a, params)
            end = done[0] & ~finished
            final = jax.tree.map(lambda new, old: jnp.where(end, new, old), info, final)
            s2 = jax.tree.map(lambda new, old: jnp.where(finished, old, new), s2, s)
            return (obs2, s2, finished | done[0], final), None

        _, _, _, _, info0 = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(env.num_agents, jnp.int32), params)
        init = (obs, s, jnp.bool_(False), jax.tree.map(jnp.zeros_like, info0))
        (_, _, _, final), _ = jax.lax.scan(body, init, jax.random.split(k_run, params.n_steps))
        keep = ("race_rank", "race_finished", "finish_time", "legs_completed", "fouls", "contacts", "shadow_time",
                "mark1_place", "team_win", "team_score")
        return {k: final[k] for k in keep if k in final}

    return jax.jit(jax.vmap(one))(keys, lineups)


def bootstrap_ci(per_race_values, n=2000, seed=0):
    """95% CI of the mean of a per-race statistic (NaN = agent absent from that race)."""
    v = np.asarray(per_race_values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(names, lineups, out):
    place = np.asarray(out["race_rank"]) + 1.0  # race_rank is 0 for the winner: place 1 = winner
    fin = np.asarray(out["race_finished"]).astype(float)
    t = np.where(fin > 0, np.asarray(out["finish_time"]), np.nan)
    fouls, contacts = np.asarray(out["fouls"], float), np.asarray(out["contacts"], float)
    rows = {}
    for a, name in enumerate(names):
        m = lineups == a
        per_race_place = np.where(m.any(1), np.nanmean(np.where(m, place, np.nan), axis=1), np.nan)
        lo, hi = bootstrap_ci(per_race_place)
        rows[name] = {
            "boats": int(m.sum()), "place": float(np.nanmean(per_race_place)), "place_ci95": [lo, hi],
            "finished": float(fin[m].mean()),
            "finish_time": float(np.nanmean(t[m])) if np.isfinite(t[m]).any() else None,
            "penalties_per_race": float(fouls[m].mean()), "contacts_per_race": float(contacts[m].mean()),
        }
    k = len(names)
    beats = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            wins = games = 0
            for r in range(len(lineups)):
                pa, pb = place[r][lineups[r] == a], place[r][lineups[r] == b]
                if len(pa) and len(pb):
                    wins += (pa[:, None] < pb[None, :]).sum() + 0.5 * (pa[:, None] == pb[None, :]).sum()
                    games += len(pa) * len(pb)
            beats[a, b] = wins / games if games else np.nan
    return rows, beats


def figure(names, rows, beats, title, path):
    order = sorted(range(len(names)), key=lambda i: rows[names[i]]["place"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.2]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for yy, i in zip(y, order):
        r = rows[names[i]]
        lo, hi = r["place_ci95"]
        ax.errorbar(r["place"], yy, xerr=[[r["place"] - lo], [hi - r["place"]]], fmt="o", color="#2c69ad", capsize=4)
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_xlim(1, BOATS)
    ax.invert_xaxis()
    ax.set_xlabel("mean place (1 = winner), 95% CI")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Ranking", fontsize=10)
    ax = axes[1]
    fins = [rows[names[i]]["finished"] for i in order]
    ax.barh(y, fins, color="#57a773")
    for yy, i in zip(y, order):
        r = rows[names[i]]
        txt = f"{100 * r['finished']:.0f}%" + (f"  {r['finish_time']:.0f} s" if r["finish_time"] else "")
        ax.text(0.02, yy, txt, va="center", fontsize=8, color="white" if r["finished"] > 0.25 else "black")
    ax.set_yticks(y, [])
    ax.set_xlim(0, 1)
    ax.set_xlabel("races finished (and mean time)")
    ax.set_title("Finishing", fontsize=10)
    ax = axes[2]
    b = beats[np.ix_(order, order)]
    im = ax.imshow(b, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    for i in range(len(order)):
        for j in range(len(order)):
            if np.isfinite(b[i, j]):
                ax.text(j, i, f"{b[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("P(row beats column)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def team_figure(names, rows, beats, title, path):
    order = sorted(range(len(names)), key=lambda i: -rows[names[i]]["team_win_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for yy, i in zip(y, order):
        r = rows[names[i]]
        lo, hi = r["ci95"]
        ax.errorbar(r["team_win_rate"], yy, xerr=[[r["team_win_rate"] - lo], [hi - r["team_win_rate"]]], fmt="o",
                    color="#2c69ad", capsize=4)
    ax.axvline(0.5, color="#999999", lw=1, ls=":")
    ax.set_yticks(y, [names[i] for i in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of team races won, 95% CI")
    ax.set_title("Team wins", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax = axes[1]
    b = beats[np.ix_(order, order)]
    im = ax.imshow(b, cmap="RdBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    for i in range(len(order)):
        for j in range(len(order)):
            if np.isfinite(b[i, j]):
                ax.text(j, i, f"{b[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("P(row's team beats column's team)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    global BOATS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agents", nargs="+", help="name=spec, e.g. A=runs/train/race_a_shaped_<time>  digital=digital")
    ap.add_argument("--env", default="race-weather", help="race preset")
    ap.add_argument("--boats", type=int, default=BOATS, help="boats per race (fleet_size)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env_params overrides, e.g. rrs16=true")
    ap.add_argument("--races", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/tournament/latest"))
    ap.add_argument("--gif", action="store_true", help="also render race 0 as a GIF")
    ap.add_argument("--teams", action="store_true", help="2v2 team racing (team_size = boats / 2): agent vs agent")
    args = ap.parse_args()

    names, specs = zip(*(a.split("=", 1) for a in args.agents))
    BOATS = args.boats
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    if args.teams:
        overrides["team_size"] = BOATS // 2
    env, params = se.make(args.env, fleet_size=BOATS, **overrides)
    assert env.num_agents == BOATS > 1, "a tournament needs a fleet race preset"
    policies = [make_agent(s)(env, params) for s in specs]
    key = jax.random.PRNGKey(args.seed)
    lineups = (team_line_ups(len(names), args.races, key, BOATS // 2) if args.teams
               else line_ups(len(names), args.races, key))
    keys = jax.random.split(jax.random.fold_in(key, 1), args.races)
    print(f"{args.races} races on {args.env}, {len(names)} agents: {', '.join(names)}")
    out = jax.tree.map(np.asarray, race_outcomes(env, params, policies, keys, jnp.asarray(lineups)))
    if args.teams:
        rows, beats = team_score(list(names), lineups, out, BOATS // 2)
        args.out.mkdir(parents=True, exist_ok=True)
        result = {"env": args.env, "boats": BOATS, "teams": True, "overrides": overrides, "races": args.races,
                  "agents": dict(zip(names, specs)), "scores": rows,
                  "beats": {names[a]: {names[b]: (None if np.isnan(beats[a, b]) else float(beats[a, b]))
                                       for b in range(len(names))} for a in range(len(names))}}
        (args.out / "tournament.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        team_figure(list(names), rows, beats, f"Team racing {BOATS // 2}v{BOATS // 2} on {args.env}: {args.races} races",
                    args.out / "tournament.png")
        print(f"{'agent':14s} {'races':>5s} {'team wins':>9s}  {'95% CI':>13s} {'finished':>8s}")
        for n in sorted(names, key=lambda n: -rows[n]["team_win_rate"]):
            r = rows[n]
            print(f"{n:14s} {r['team_races']:5d} {100 * r['team_win_rate']:8.0f}%  [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] "
                  f"{100 * r['finished']:7.0f}%")
        print(f"wrote {args.out}/tournament.json and tournament.png")
        return
    rows, beats = score(list(names), lineups, out)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"env": args.env, "boats": BOATS, "overrides": overrides, "races": args.races, "seed": args.seed, "agents": dict(zip(names, specs)),
              "scores": rows, "beats": {names[a]: {names[b]: (None if np.isnan(beats[a, b]) else float(beats[a, b]))
                                                     for b in range(len(names))} for a in range(len(names))}}
    (args.out / "tournament.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    title = f"Tournament on {args.env}: {args.races} races, {BOATS} boats each" + (f" ({args.set and ' '.join(args.set)})" if args.set else "")
    figure(list(names), rows, beats, title, args.out / "tournament.png")
    print(f"{'agent':14s} {'boats':>5s} {'place':>6s}  {'95% CI':>13s} {'finished':>8s} {'time':>6s} {'pen/race':>8s}")
    for n in sorted(names, key=lambda n: rows[n]["place"]):
        r = rows[n]
        t = f"{r['finish_time']:.0f}" if r["finish_time"] else "-"
        print(f"{n:14s} {r['boats']:5d} {r['place']:6.2f}  [{r['place_ci95'][0]:.2f}, {r['place_ci95'][1]:.2f}] "
              f"{100 * r['finished']:7.0f}% {t:>6s} {r['penalties_per_race']:8.2f}")
    print(f"wrote {args.out}/tournament.json and tournament.png")

    if args.gif:  # race 0, rendered with the course
        from sail_env.runs import params_dict, render_replay, run_episode

        record, _ = jax.jit(lambda k: run_episode(env, params, k, policies, jnp.asarray(lineups[0])))(keys[0])
        record = jax.tree.map(np.asarray, record)
        lineup_txt = ", ".join(f"{i}:{names[a]}" for i, a in enumerate(lineups[0]))
        render_replay({"record": record, "params": params_dict(params), "name": lineup_txt},
                      args.out / "race0.gif", title=f"race 0 ({lineup_txt})")
        print(f"wrote {args.out}/race0.gif")


if __name__ == "__main__":
    main()
