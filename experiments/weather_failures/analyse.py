# Step 2: which weather features separate failing races from finishing ones (AUC), and how stuck boats sail.
#   python experiments/weather_failures/analyse.py runs/weather_failures.pkl
import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], "rb")); res, feat = d["res"], d["feat"]
pols = [k for k in res if k != "digital"]
fails = np.array([~np.isfinite(res[k]["finish"]) for k in pols])  # (6, N)
nf = fails.sum(0)
print("races failed by k of 6 policies:", {k: int((nf == k).sum()) for k in range(7)})
bad, good = nf >= 2, nf == 0
print(f"\n{'feature':14s} {'fail(>=2) mean':>14s} {'finish mean':>12s}  AUC")
for k, v in feat.items():
    a, b = v[bad], v[good]
    auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
    print(f"{k:14s} {a.mean():14.3f} {b.mean():12.3f}  {auc:.2f}")
# Where and how: per failure, the last 300 s before the time limit vs normal beating
print("\nstuck boats (last 300 s alive) vs boats on a normal beat:")
st, nb = dict(ws=[], spd=[], twa=[], turn=[], leg=[]), dict(ws=[], spd=[], twa=[], turn=[])
for i, k in enumerate(pols):
    r = res[k]
    twa = np.degrees(np.abs((r["hdg"] - r["wd"] + np.pi) % (2 * np.pi) - np.pi))
    turn = np.degrees(np.abs((np.diff(r["hdg"], axis=1) + np.pi) % (2 * np.pi) - np.pi))
    for j in np.where(fails[i])[0]:
        live = np.where(~r["ended"][j])[0][-300:]
        st["ws"].append(r["ws"][j, live].mean()); st["spd"].append(r["spd"][j, live].mean())
        st["twa"].append(np.median(twa[j, live])); st["turn"].append(turn[j, live[:-1]].mean()); st["leg"].append(r["leg"][j, live[-1]])
    for j in np.where(good)[0][:40]:
        on = np.where(~r["ended"][j] & (r["leg"][j] == 0))[0][60:]
        nb["ws"].append(r["ws"][j, on].mean()); nb["spd"].append(r["spd"][j, on].mean())
        nb["twa"].append(np.median(twa[j, on])); nb["turn"].append(turn[j, on[:-1]].mean())
for k in ["ws", "spd", "twa", "turn"]:
    print(f"  {k:5s} stuck {np.mean(st[k]):6.2f}  normal beat {np.mean(nb[k]):6.2f}")
print("  stuck on leg:", np.bincount(st["leg"], minlength=4))
print("  share of stuck boats with local wind < 3.5 m/s:", np.mean(np.array(st["ws"]) < 3.5))
