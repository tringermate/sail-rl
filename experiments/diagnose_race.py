"""Where does a trained racer lose time against the digital racer? Same starts, leg by leg.

    uv run python experiments/diagnose_race.py runs/train/ladder_s3_beat_run_<stamp> [--races 64] [--out fig.png]

Races every seed of the run (greedy) and the digital racer from the same start-bank states (the run's own
start bank: same seed, so the starts the run trained on), on the run's course (its env_params: race_legs,
beat length, gate). Per agent it reports:
    leg times      time spent on each leg (s)
    overstand      on each beat, how far upwind of the windward mark the boat got before rounding (m):
                   ~0-10 m = rounded off the layline, more = sailed past the layline and came back
    tacks / gybes  changes of tack on the beats / on the runs (per leg)
    run angle      |true wind angle| sailed on the runs (deg; 180 = dead downwind)
and saves a figure: leg-time bars, overstand, run-angle histogram and one race's tracks.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import sail_env as se  # noqa: E402
from sail_env.agents import digital_sailor_agent, pretrained_agent  # noqa: E402
from sail_env.race_env import course_geometry  # noqa: E402
from sail_env.train.curriculum.config_env import make_race_start_bank  # noqa: E402

LEG_NAMES = ["beat 1", "run 1", "beat 2", "run 2"]


def sail(env, params, policy, bank, steps):
    """Race every start in `bank` with `policy`; per-step tracks (races, steps) and the finish times."""
    n = env.num_agents

    def race(s):
        def body(c, i):
            o, s, fin = c
            a = jax.vmap(policy, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(i), o, s.boats, jnp.arange(n))
            o2, s2, _, d, _ = env.step_env(jax.random.PRNGKey(0), s, a, params)
            s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)  # freeze after the end
            b = s.boats.boat
            rec = (b.x[0], b.y[0], b.heading[0], s.leg[0], fin)
            return (o2, s2, fin | d[0]), rec

        (_, s_end, _), rec = jax.lax.scan(body, (env.get_obs(s, params), s, jnp.bool_(False)), jnp.arange(steps))
        return rec, s_end.finish_time[0]

    rec, ft = jax.jit(jax.vmap(race))(bank)
    x, y, hdg, leg, ended = (np.asarray(r) for r in rec)
    return dict(x=x, y=y, hdg=hdg, leg=leg, ended=ended, finish=np.asarray(ft))


def analyse(tr, k, wind_dir, legs):
    """Leg times, overstand, tacks/gybes and run angles from the tracks of one agent."""
    leg, x, y = tr["leg"], tr["x"], tr["y"]
    twa = (tr["hdg"] - wind_dir[:, None] + np.pi) % (2 * np.pi) - np.pi  # signed, + = starboard... side
    side = np.sign(twa)
    races, steps = leg.shape
    t_leg = np.full((races, legs), np.nan)  # seconds on each leg (nan if never completed)
    over = []  # per beat: max distance upwind of the mark before rounding
    changes = np.zeros((races, legs))
    run_angles, beat_angles, beat_vmg = [], [], []
    wx, wy = k["w"]
    ux, uy = k["u"]
    for r in range(races):
        live = ~tr["ended"][r]
        for j in range(legs):
            on = live & (leg[r] == j)
            if not on.any():
                continue
            if (leg[r] > j).any():  # completed this leg
                t_leg[r, j] = on.sum()
            s = side[r][on]
            changes[r, j] = np.count_nonzero(np.diff(s[s != 0]))
            if j % 2 == 0:  # beat: how far upwind of the mark
                up = (x[r][on] - wx[r]) * ux[r] + (y[r][on] - wy[r]) * uy[r]
                if (leg[r] > j).any():
                    over.append(max(up.max(), 0.0))
                beat_angles.append(np.degrees(np.abs(twa[r][on])))
                beat_vmg.append(np.diff(up))  # m/s made good upwind, per second
            else:
                run_angles.append(np.degrees(np.abs(twa[r][on])))
    return dict(t_leg=t_leg, overstand=np.array(over), changes=changes,
                beat_angles=np.concatenate(beat_angles), beat_vmg=np.concatenate(beat_vmg),
                run_angles=np.concatenate(run_angles) if run_angles else np.array([]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--races", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500, help="seconds of racing at most")
    ap.add_argument("--out", type=Path, default=Path("runs") / "diagnose_race.png")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="env_params overrides for the races")
    args = ap.parse_args()

    cfg = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    overrides = dict(cfg.get("env_params") or {})
    for kv in args.set:  # race in other conditions than the run trained in, e.g. current_max_frac=0.06
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    env, params = se.make(cfg["env"], **overrides)
    legs = params.race_legs
    sb = cfg.get("start_bank", {})
    bank = make_race_start_bank(env, params, jax.random.PRNGKey(sb.get("seed", 4321)), args.races, sb.get("policy", "fleet"))

    # Course and wind of every race (no weather: the base wind is the wind everywhere).
    line = jax.tree.map(lambda a: a[:, 0], bank.boats.line)
    wind_dir = np.asarray(bank.boats.wind.base_dir[:, 0])
    k = jax.vmap(course_geometry, in_axes=(0, 0, None))(line, jnp.asarray(wind_dir), params)
    k = jax.tree.map(np.asarray, k)

    agents = {f"seed {s}": pretrained_agent(str(args.run), seed=s)(env, params) for s in range(cfg["ppo"]["num_seeds"])}
    agents["digital"] = digital_sailor_agent(env, params)
    res = {}
    for name, pol in agents.items():
        tr = sail(env, params, pol, bank, args.steps)
        a = analyse(tr, k, wind_dir, legs)
        a["tracks"] = tr
        res[name] = a
        fin = np.isfinite(tr["finish"])
        legs_txt = "  ".join(f"{LEG_NAMES[j]} {np.nanmean(a['t_leg'][:, j]):5.1f}" for j in range(legs))
        chg = "  ".join(f"{np.mean(a['changes'][:, j]):.1f}" for j in range(legs))
        print(f"{name:8s} finished {fin.mean():4.0%}  time {np.nanmean(np.where(fin, tr['finish'], np.nan)):6.1f} s | "
              f"{legs_txt} | beat |TWA| median {np.median(a['beat_angles']):4.1f} deg, VMG {np.mean(a['beat_vmg']):.2f} m/s | overstand {np.median(a['overstand']):5.1f} m (median) | tack/gybe per leg {chg} | "
              f"run |TWA| median {np.median(a['run_angles']) if a['run_angles'].size else float('nan'):5.1f} deg")

    # Fair score: every race counts, an unfinished race scores the race's time limit (t_end, s after the
    # gun), as a DNF. Paired with the digital racer race by race (same start), bootstrap 95% interval.
    dnf = float(params.t_end)
    print(f"\nall races, unfinished = {dnf:.0f} s (time limit); difference vs digital (negative = faster), 95% bootstrap interval")
    scored = {n: np.where(np.isfinite(r["tracks"]["finish"]), r["tracks"]["finish"], dnf) for n, r in res.items()}
    rng = np.random.default_rng(0)
    for n, t in scored.items():
        line = f"{n:8s} all-races time {t.mean():6.1f} s"
        if n != "digital":
            d = t - scored["digital"]
            boot = rng.choice(d, (10000, d.size)).mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            line += (f" | vs digital {d.mean():+6.1f} s  [{lo:+6.1f}, {hi:+6.1f}] | "
                     f"faster in {np.mean(d < 0):4.0%} of races")
        print(line)

    # ---- figure ----------------------------------------------------------------------------------
    names = list(res)
    colors = {n: c for n, c in zip(names, ["#9fc5e8", "#2c69ad", "#e69138", "#333333"][-len(names):])}
    colors["digital"] = "#333333"
    fig, ax = plt.subplots(1, 5, figsize=(21, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1.2, 1.2, 1]})
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax[0].bar(np.arange(legs) + i * w, np.nanmean(res[n]["t_leg"], axis=0), w, color=colors[n], label=n)
    ax[0].set_xticks(np.arange(legs) + 0.4 - w / 2, LEG_NAMES[:legs]); ax[0].set_ylabel("s"); ax[0].legend(fontsize=8)
    ax[0].set_title("time per leg (mean)", fontsize=10)
    ax[1].boxplot([res[n]["overstand"] for n in names], tick_labels=names, showfliers=False)
    ax[1].set_title("overstand: m upwind of the mark before rounding", fontsize=10)
    bins = np.linspace(90, 180, 31)
    for n in names:
        if res[n]["run_angles"].size:
            ax[2].hist(res[n]["run_angles"], bins, histtype="step", lw=1.8, color=colors[n], density=True, label=n)
    ax[2].set_xlabel("|true wind angle| on the runs (deg, 180 = dead downwind)"); ax[2].legend(fontsize=8)
    ax[2].set_title("angles sailed downwind", fontsize=10)
    for n in names:
        ax[3].hist(res[n]["beat_angles"], np.linspace(0, 90, 46), histtype="step", lw=1.8, color=colors[n], density=True,
                   label=f"{n}: VMG {np.mean(res[n]['beat_vmg']):.2f} m/s")
    ax[3].set_xlabel("|true wind angle| on the beats (deg)"); ax[3].legend(fontsize=8)
    ax[3].set_title("angles sailed upwind (incl. tacks)", fontsize=10)
    best = min((n for n in names if n != "digital"), key=lambda n: np.nanmean(res[n]["t_leg"].sum(1)))
    for n in (best, "digital"):
        tr = res[n]["tracks"]
        live = ~tr["ended"][0]
        ax[4].plot(tr["x"][0][live], tr["y"][0][live], color=colors[n], lw=1.5, label=f"{n}: {tr['finish'][0]:.0f} s")
    ax[4].plot([k["w"][0][0]], [k["w"][1][0]], "o", color="gold", mec="k")  # windward mark of race 1
    ax[4].plot([k["g"][0][0]], [k["g"][1][0]], "s", color="orange", mec="k")  # gate centre
    ax[4].set_aspect("equal"); ax[4].legend(fontsize=8); ax[4].set_title("race 1: tracks", fontsize=10)
    fig.suptitle(f"{args.run.name}: {args.races} races from the same starts", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print("figure ->", args.out)


if __name__ == "__main__":
    main()
