"""Run training configs one after another in the background, each followed by the race diagnosis.

    uv run python experiments/queue.py [--wait-for <pid>] cfg1.yaml:init cfg2.yaml:init ...

Each item is a config path, optionally followed by ':' and where to start from:
    :runs/train/<run>     a run directory
    :latest=<name>        the newest finished run called <name> (runs/train/<name>_<stamp>, with metrics.npz),
                          so a later item can start from an earlier item of the same queue
    (nothing)             init_from as in the config, or from scratch
A run that fails is logged and the queue goes on (items depending on it are skipped). After every finished
run, experiments/diagnose_race.py writes runs/diag/<run>.png (leg times, angles vs the digital racer).
Progress: runs/queue.log, one line per event; each run's console output in runs/<name>.log.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

LOG = Path("runs") / "queue.log"


def note(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M')}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest(name):
    runs = sorted(d for d in Path("runs/train").glob(f"{name}_2*") if (d / "metrics.npz").exists())
    return runs[-1] if runs else None


def pid_alive(pid):
    if os.name != "nt":  # Linux (a GPU pod): signal 0 only checks that the process exists
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--wait-for", type=int, help="start only when this process (a running training) has ended")
    args = ap.parse_args()

    if args.wait_for:
        note(f"queue: waiting for process {args.wait_for} to finish")
        while pid_alive(args.wait_for):
            time.sleep(60)
    for item in args.items:
        cfg_path, _, init = item.partition(":")
        if len(cfg_path) == 1:  # a Windows drive letter, not a separator: C:/...
            cfg_path, _, init = item[2:].partition(":")
            cfg_path = item[:2] + cfg_path
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        name = cfg["name"]
        if init.startswith("latest="):
            src = latest(init[len("latest="):])
            if src is None:
                note(f"{name}: SKIPPED, no finished run {init[len('latest='):]!r} to start from")
                continue
            cfg["init_from"] = str(src)
        elif init:
            cfg["init_from"] = init
        tmp = Path("runs") / "queue_configs" / f"{name}.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        note(f"{name}: training {cfg['env']}, {cfg['ppo']['total_timesteps']:,} steps per seed, "
             f"from {cfg.get('init_from', 'scratch')}" + (f", teacher {cfg['teacher']}" if cfg.get("teacher") else ""))
        with (Path("runs") / f"{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-m", "sail_env.train.run", str(tmp)], stdout=log,
                                  stderr=subprocess.STDOUT)
        text = (Path("runs") / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        found = re.search(r"->\s+(runs\S+)", text)
        run_dir = Path(found.group(1)) if found else None
        if proc.returncode != 0 or run_dir is None or not (run_dir / "metrics.npz").exists():
            note(f"{name}: FAILED (exit {proc.returncode}); see runs/{name}.log"
                 + (f" (checkpoint in {run_dir})" if run_dir else ""))
            continue
        note(f"{name}: finished -> {run_dir}")
        diag = Path("runs") / "diag" / f"{run_dir.name}.png"
        d = subprocess.run([sys.executable, "experiments/diagnose_race.py", str(run_dir), "--steps", "1800",
                            "--out", str(diag)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        summary = [ln for ln in d.stdout.splitlines() if "finished" in ln]
        (diag.with_suffix(".txt")).write_text(d.stdout + d.stderr, encoding="utf-8")
        note(f"{name}: diagnosis -> {diag}" if d.returncode == 0 else f"{name}: diagnosis FAILED, see {diag.with_suffix('.txt')}")
        for ln in summary:
            note("    " + ln[:150])
    note("queue: done")


if __name__ == "__main__":
    main()
