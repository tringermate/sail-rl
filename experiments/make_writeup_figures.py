"""Figures for docs/writeup_racing.md, from the saved runs (runs/train/*/metrics.npz) and race diagnoses
(runs/diag/*.txt, written by experiments/diagnose_race.py).

    uv run python experiments/make_writeup_figures.py          -> docs/figures/racing/*.png

Curves are training episodes (the policy samples its actions); the race diagnoses are greedy races from the
start bank, paired with the digital racer, and are the numbers quoted in the text.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path("runs/train")
DIAG = Path("runs/diag")
OUT = Path("docs/figures/racing")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4", "#333333"]


def curve(run, key, smooth=0.02, skip=0.03):
    """(steps per seed, mean over seeds) of a metric, smoothed over a moving window of updates. The first
    `skip` of the updates are left out: in the first updates only the shortest episodes have ended."""
    m = np.load(RUNS / run / "metrics.npz")
    x = np.asarray(m["steps"]).reshape(-1)[: m[key].shape[-1]]
    y = np.nanmean(np.asarray(m[key], dtype=float), axis=0)
    w = max(1, int(len(y) * smooth))
    ys = np.array([np.nanmean(y[max(0, i - w + 1): i + 1]) for i in range(len(y))])
    k = int(len(y) * skip)
    return x[k:], ys[k:]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e3e3e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def ladder():
    stages = [("ladder_s1_beat100_20261024-095647", "1: beat 100 m"), ("ladder_s2_beat400_20261024-101948", "2: beat 400 m"),
              ("ladder_s3_beat_run_20261024-134723", "3: beat + run"), ("ladder_s4_full_20261024-170950", "4: full race")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    offset = 0
    for i, (run, name) in enumerate(stages):
        for a, key in zip(ax, ["race_finished_rate", "finish_time"]):
            x, y = curve(run, key)
            a.plot((x + offset) / 1e6, y, color=COLORS[i], lw=2, label=f"stage {name}")
        offset += x[-1]
        for a in ax:
            a.axvline(offset / 1e6, color="#bbbbbb", lw=0.8, ls=":")
    style(ax[0], "Racing ladder: share of races finished (training)", "steps per seed, cumulative (millions)", "finished")
    style(ax[1], "Racing ladder: finish time of finished races (training)", "steps per seed, cumulative (millions)", "s")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ladder_curves.png", dpi=110)


def kickstart():
    """Stage 4 full race: pure PPO vs kickstarted (laptop) vs kickstarted from scratch vs kickstarted on GPU."""
    lines = [("ladder_s4_full_20261024-170950", 0, "pure PPO from stage 3 (laptop, 64 envs)", COLORS[7]),
             ("ladder_s4k_resume_20261025-073036", 13_107_200, "kickstarted from stage 3 (laptop; first 13M not saved)", COLORS[0]),
             ("a1_kick_from_scratch_20261025-132443", 0, "kickstarted from scratch (laptop)", COLORS[2]),
             ("gpu_s4k_2048_20261026-095733", 0, "kickstarted from stage 3 (RTX 4090, 2048 envs)", COLORS[1])]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for run, off, name, c in lines:
        for a, key in zip(ax, ["race_finished_rate", "finish_time"]):
            x, y = curve(run, key)
            a.plot((x + off) / 1e6, y, color=c, lw=2, label=name)
    for a in ax:
        a.set_xscale("log")
    ax[1].axhline(672.7, color=COLORS[7], lw=1, ls="--")
    ax[1].text(1.2, 676, "digital racer (greedy races, 672.7 s)", fontsize=8, color="#555555")
    ax[1].set_ylim(640, 900)
    style(ax[0], "Full race: share finished (training)", "steps per seed (millions, log)", "finished")
    style(ax[1], "Full race: finish time (training, sampled actions)", "steps per seed (millions, log)", "s")
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "kickstart_curves.png", dpi=110)


def weather():
    lines = [("w1_weather_from_kick_20261025-085711", 0, "W1: from kickstarted racer", COLORS[0]),
             ("w2_weather_from_pure_20261025-105629", 0, "W2: from pure-PPO racer", COLORS[7]),
             ("w3_weather_kick_20261025-155059", 0, "W3: with the digital teacher", COLORS[3]),
             ("w4_capped_from_w1_20261026-080754", 0, "W4: current capped (laptop, to 13M)", COLORS[4]),
             ("gpu_w4_resume_20261026-094008", 13_107_200, "W4 resumed (GPU, 64 envs)", COLORS[4]),
             ("gpu_w5a_weather_from_w4_20261026-103537", 0, "W5a: from W4, 2048 envs (GPU)", COLORS[2]),
             ("gpu_w5b_weather_from_s4k_20261026-105810", 0, "W5b: from GPU racer, 2048 envs (GPU)", COLORS[1])]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    seen = set()
    for run, off, name, c in lines:
        if not (RUNS / run / "metrics.npz").exists():
            continue
        for a, key in zip(ax, ["race_finished_rate", "finish_time"]):
            x, y = curve(run, key)
            a.plot((x + off) / 1e6, y, color=c, lw=2, label=None if name.split(":")[0] in seen else name)
        seen.add(name.split(":")[0])
    for a in ax:
        a.set_xscale("log")
    ax[0].set_ylim(0.85, 1.005)
    ax[1].set_ylim(660, 800)
    style(ax[0], "Weather race: share finished (training)", "steps per seed (millions, log)", "finished")
    style(ax[1], "Weather race: finish time (training, sampled actions)", "steps per seed (millions, log)", "s")
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "weather_curves.png", dpi=110)


def fair_scores():
    """Paired difference to the digital racer (all races, DNF = time limit) with 95% bootstrap intervals."""
    agents = [("ladder_s4_full_20261024-170950.txt", "pure PPO (laptop)"),
              ("ladder_s4k_resume_20261025-073036.txt", "kickstarted (laptop)"),
              ("a1_kick_from_scratch_20261025-132443.txt", "kickstarted from scratch"),
              ("gpu_s4k_2048_20261026-095733.txt", "kickstarted (GPU, 2048 envs)"),
              (None, None),
              ("w1_weather_from_kick_20261025-085711_capped.txt", "W1 weather (256 races)"),
              ("w2_weather_from_pure_20261025-105629.txt", "W2 weather (from pure; uncapped current)"),
              ("w3_weather_kick_20261025-155059_capped.txt", "W3 weather, teacher (256 races)"),
              ("gpu_w4_resume.txt", "W4 weather, trained capped (256 races)"),
              ("gpu_w5a_weather_from_w4_20261026-103537.txt", "W5a weather (GPU)"),
              ("gpu_w5b_256.txt", "W5b weather (GPU), 256 races")]
    pat = re.compile(r"seed (\d)\s+all-races time\s+([\d.]+) s \| vs digital\s+([+-][\d.]+) s\s+\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y, labels = 0, []
    for f, name in agents:
        if f is None:
            y -= 0.6
            continue
        if not (DIAG / f).exists():
            continue
        rows = [m.groups() for m in pat.finditer((DIAG / f).read_text(encoding="utf-8", errors="replace"))]
        for s, _, d, lo, hi in rows:
            yy = y - 0.22 * (int(s) - 1)
            ax.errorbar(float(d), yy, xerr=[[float(d) - float(lo)], [float(hi) - float(d)]], fmt="o", ms=6,
                        color=COLORS[int(s)], capsize=3, lw=1.5, label=f"seed {s}" if not labels else None)
        labels.append((y, name))
        y -= 1
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks([t for t, _ in labels], [n for _, n in labels])
    ax.text(1, labels[0][0] + 0.6, "slower than digital ->", fontsize=8, color="#555555")
    ax.text(-1, labels[0][0] + 0.6, "<- faster", fontsize=8, color="#555555", ha="right")
    style(ax, "Paired race time vs the digital racer, 95% bootstrap intervals\n"
              "DNF = time limit; steady wind 64 races; weather: current capped unless noted",
          "difference to the digital racer (s), same starts", "")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fair_scores.png", dpi=110)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (ladder, kickstart, weather, fair_scores):
        f()
        print(f.__name__, "done")
