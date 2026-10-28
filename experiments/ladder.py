"""Train racers in stages: each stage starts from the previous stage's network and makes the task harder.

    uv run python experiments/ladder.py experiments/configs/ladder/s1_beat100.yaml \\
        experiments/configs/ladder/s2_beat400.yaml ... [--gate 0.8] [--from runs/train/<run>]

For every stage: write its config with init_from = the previous stage's run (or --from for the first),
train it (python -m sail_env.train.run, W&B as in the config), then check the gate: the share of races
finished over the last 5% of updates (mean of the seeds). Below the gate the ladder stops, so a stage that
did not learn is never built on. Progress goes to runs/ladder.log, one line per stage.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

LOG = Path("runs") / "ladder.log"


def note(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def finish_rate(run_dir: Path):
    """Share of races finished over the last 5% of updates, mean over seeds (training episodes)."""
    m = np.load(run_dir / "metrics.npz")
    x = m["race_finished_rate"]
    tail = x[:, -max(x.shape[1] // 20, 1):]
    return float(np.nanmean(tail))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", type=Path)
    ap.add_argument("--gate", type=float, default=0.8)
    ap.add_argument("--from", dest="start_from", type=Path, help="initialise the first stage from this run")
    args = ap.parse_args()

    prev = args.start_from
    for cfg_path in args.stages:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if prev is not None:
            cfg["init_from"] = str(prev)
        tmp = Path("runs") / "ladder_configs" / cfg_path.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        note(f"{cfg['name']}: training ({cfg['ppo']['total_timesteps']:,} steps per seed)"
             + (f", from {prev}" if prev else ", from scratch"))
        proc = subprocess.run([sys.executable, "-m", "sail_env.train.run", str(tmp)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        (Path("runs") / f"{cfg['name']}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        found = re.search(r"->\s+(\S+)", proc.stdout)
        if proc.returncode != 0 or not found:
            note(f"{cfg['name']}: FAILED (exit {proc.returncode}); see runs/{cfg['name']}.log")
            return 1
        run_dir = Path(found.group(1))
        rate = finish_rate(run_dir)
        note(f"{cfg['name']}: finished {100 * rate:.0f}% of races (gate {100 * args.gate:.0f}%) -> {run_dir}")
        if not rate >= args.gate:
            note(f"STOPPED at {cfg['name']}: below the gate")
            return 2
        prev = run_dir
    note("LADDER COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
