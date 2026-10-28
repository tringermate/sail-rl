# Step 3: the figure (failures vs wind speed and current, one stuck race).
#   python experiments/weather_failures/plot.py runs/weather_failures.pkl docs/figures/racing/stuck_cause.png
import pickle, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pickle.load(open(sys.argv[1], "rb")); res, feat = d["res"], d["feat"]
pols = [k for k in res if k != "digital"]
nf = np.array([~np.isfinite(res[k]["finish"]) for k in pols]).sum(0)
cur, tws = feat["current"], feat["tws"]
ratio = cur / tws
bad = nf >= 2
for thr in [0.04, 0.06, 0.08]:
    m = ratio > thr
    print(f"current/tws > {thr}: {m.mean():4.0%} of races, failure rate there {np.mean(nf[m] >= 2):4.0%}, elsewhere {np.mean(nf[~m] >= 2):4.1%}")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
sc = ax[0].scatter(tws, cur, c=nf, cmap="Reds", vmin=0, vmax=6, s=18, edgecolor="k", lw=0.3)
ax[0].set_xlabel("wind speed (m/s)"); ax[0].set_ylabel("current (m/s)")
ax[0].set_title("256 races: how many of 6 policies fail", fontsize=10); fig.colorbar(sc, ax=ax[0])
xs = np.linspace(tws.min(), tws.max(), 10); ax[0].plot(xs, 0.06 * xs, "b--", lw=1, label="current = 6% of wind"); ax[0].legend(fontsize=8)
# a stuck race and its tracks
j = int(np.where(nf == 6)[0][0]); r = res["w1 s1"]; dg = res["digital"]
for src, c, lab in [(dg, "#333", "digital"), (r, "#e69138", "W1 seed 1")]:
    live = ~src["ended"][j]
    ax[1].plot(src["x"][j][live], src["y"][j][live], color=c, lw=0.8, label=lab)
ax[1].set_aspect("equal"); ax[1].legend(fontsize=8)
ax[1].set_title(f"race {j}: wind {tws[j]:.1f} m/s, current {cur[j]:.2f} m/s", fontsize=10)
twa = np.degrees(np.abs((r["hdg"][j] - r["wd"][j] + np.pi) % (2 * np.pi) - np.pi))
live = ~r["ended"][j]
ax[2].plot(np.arange(live.sum()), twa[live], lw=0.5, color="#e69138")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("|TWA| (deg)"); ax[2].set_title(f"race {j}, W1 seed 1: angle to the wind", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=90)
