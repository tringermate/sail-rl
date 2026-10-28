# Step 1 of the weather-failure analysis (writeup_racing.md, 5.3): race W1 and W3 (3 seeds each) and the digital
# racer over 256 starts, logging the wind at the boat; saves a pickle for analyse.py and plot.py.
#   python experiments/weather_failures/race.py runs/weather_failures.pkl
"""Why do weather racers get lost? Log the wind at the boat, compare failing and finishing races."""
import sys, pickle
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, yaml
import sail_env as se
from sail_env.agents import pretrained_agent, digital_sailor_agent
from sail_env.train.curriculum.config_env import make_race_start_bank
RUNS = ["w1_weather_from_kick_20261025-085711", "w3_weather_kick_20261025-155059"]
N, STEPS = 256, 1250
cfg = yaml.safe_load(open(f"runs/train/{RUNS[0]}/config.yaml"))
env, params = se.make(cfg["env"], **(cfg.get("env_params") or {}))
bank = make_race_start_bank(env, params, jax.random.PRNGKey(4321), N, "fleet")
n = env.num_agents

def sail(policy):
    def race(s):
        def body(c, i):
            o, s, fin = c
            a = jax.vmap(policy, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(i), o, s.boats, jnp.arange(n))
            wd, ws = env.local_winds(s.boats, params)
            wd, ws = jnp.broadcast_to(wd, (n,))[0], jnp.broadcast_to(ws, (n,))[0]
            o2, s2, _, d, _ = env.step_env(jax.random.PRNGKey(0), s, a, params)
            s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)
            b = s.boats.boat
            return (o2, s2, fin | d[0]), (b.x[0], b.y[0], b.heading[0], b.speed[0], s.leg[0], fin, wd, ws, a[0])
        (_, se_, _), rec = jax.lax.scan(body, (env.get_obs(s, params), s, jnp.bool_(False)), jnp.arange(STEPS))
        return rec, se_.finish_time[0]
    rec, ft = jax.jit(jax.vmap(race))(bank)
    keys = ["x", "y", "hdg", "spd", "leg", "ended", "wd", "ws", "act"]
    out = {k: np.asarray(v) for k, v in zip(keys, rec)}
    out["finish"] = np.asarray(ft)
    return out

res = {}
for r in RUNS:
    for s in range(3):
        res[f"{r[:2]} s{s}"] = sail(pretrained_agent(f"runs/train/{r}", seed=s)(env, params))
        print(f"{r[:2]} s{s}: unfinished {np.sum(~np.isfinite(res[f'{r[:2]} s{s}']['finish']))}/{N}", flush=True)
res["digital"] = sail(digital_sailor_agent(env, params))
print("digital unfinished", np.sum(~np.isfinite(res["digital"]["finish"])))
w, wx = bank.boats.wind, bank.boats.weather
feat = dict(tws=np.asarray(w.tws)[:, 0],
            current=np.hypot(np.asarray(wx.current_x)[:, 0], np.asarray(wx.current_y)[:, 0]),
            wave_h=np.asarray(wx.wave_h)[:, 0], drift=np.asarray(wx.drift)[:, 0],
            lull=np.asarray(wx.puff_amp)[:, 0].min(-1), puff=np.asarray(wx.puff_amp)[:, 0].max(-1),
            shift=np.abs(np.asarray(wx.shift_amp)[:, 0]).sum(-1), grad=np.abs(np.asarray(wx.grad)[:, 0]),
            bend=np.abs(np.asarray(wx.bend)[:, 0]))
# current direction relative to the wind: +1 = current flows upwind (into the wind), -1 = with the wind
bd = np.asarray(w.base_dir)[:, 0]
cx, cy = np.asarray(wx.current_x)[:, 0], np.asarray(wx.current_y)[:, 0]
feat["cur_into_wind"] = (cx * np.sin(bd) + cy * np.cos(bd))  # m/s toward where the wind comes from
pickle.dump(dict(res=res, feat=feat), open(sys.argv[1], "wb"))
