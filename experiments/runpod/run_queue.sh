#!/usr/bin/env bash
# On the pod: start experiments/queue.py in the background, detached from the SSH session, with the pod's
# environment (WANDB_API_KEY, WANDB_BASE_URL, ... set in the Runpod console). Runpod writes those to
# /etc/rp_environment, which a non-interactive SSH command does not always load, and W&B then fails with
# "No API key configured".
#
#   bash /workspace/sail-rl/experiments/runpod/run_queue.sh cfg1.yaml[:init] cfg2.yaml[:init] ...
#   tail -f /workspace/sail-rl/runs/queue.log
set -euo pipefail
# Only the WANDB_* lines: the whole file also resets library paths from the base image, and JAX then cannot
# find the CUDA libraries of its own pip wheels ("Unable to load cuSPARSE") and silently falls back to CPU.
if [ -r /etc/rp_environment ]; then eval "$(grep -E '^export WANDB_[A-Z_]*=' /etc/rp_environment)"; fi
if [ -z "${WANDB_API_KEY:-}" ]; then echo "warning: WANDB_API_KEY is not set; runs with wandb: true will fail" >&2; fi
# W&B's local run folder on the container disk, not on /workspace (a network filesystem): there its sender
# stalled after the first logged GIF, and no curves reached the server after that (the local .wandb file
# kept growing; push it later with: wandb sync <run dir>).
export WANDB_DIR=/root/wandb_local
mkdir -p "$WANDB_DIR"
cd /workspace/sail-rl
mkdir -p runs
/workspace/venv/bin/python -c 'import jax; assert jax.default_backend() == "gpu", jax.devices()' \
    || { echo "JAX does not see the GPU: not starting" >&2; exit 1; }
setsid nohup /workspace/venv/bin/python experiments/queue.py "$@" > runs/queue.out 2>&1 < /dev/null &
echo "queue started (pid $!): tail -f /workspace/sail-rl/runs/queue.log"
