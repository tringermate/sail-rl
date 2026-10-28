#!/usr/bin/env bash
# Talk to the current GPU pod from the laptop: one entry point for every remote command, copy and wait, so a
# single permission rule can cover them. The pod's address lives in runs/runpod/pod.env (not in git):
#
#   POD_HOST=<pod IP>
#   POD_PORT=19689
#
# Host keys are checked against ~/.ssh/known_hosts_runpod (add a new pod's key once, after comparing its
# fingerprint with the pod's boot log); the key is ~/.ssh/id_ed25519_runpod.
#
#   bash experiments/runpod/pod.sh run '<command on the pod>'
#   bash experiments/runpod/pod.sh get <remote path> <local dir>      (recursive)
#   bash experiments/runpod/pod.sh put <local path> <remote path>
#   bash experiments/runpod/pod.sh wait <remote file> <regex> [max minutes, default 120]
#        polls every 30 s until a line of the file matches (prints the matching lines; exit 1 on timeout)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
env_file="$here/runs/runpod/pod.env"
[ -r "$env_file" ] || { echo "no $env_file (POD_HOST=..., POD_PORT=...)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$env_file"
opts=(-i "$HOME/.ssh/id_ed25519_runpod" -o "UserKnownHostsFile=$HOME/.ssh/known_hosts_runpod"
      -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 -o ServerAliveInterval=30)
target="root@$POD_HOST"

cmd="${1:-}"; shift || true
case "$cmd" in
    run)  ssh "${opts[@]}" -p "$POD_PORT" "$target" "$@" ;;
    get)  mkdir -p "$2"; scp "${opts[@]}" -P "$POD_PORT" -r "$target:$1" "$2" ;;
    put)  scp "${opts[@]}" -P "$POD_PORT" -r "$1" "$target:$2" ;;
    wait)
        file="$1"; pattern="$2"; minutes="${3:-120}"
        for _ in $(seq 1 $((minutes * 2))); do
            if out=$(ssh "${opts[@]}" -p "$POD_PORT" "$target" "grep -E '$pattern' '$file'" 2>/dev/null); then
                echo "$out"; exit 0
            fi
            sleep 30
        done
        echo "timeout after $minutes min waiting for /$pattern/ in $file" >&2; exit 1 ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
esac
