#!/usr/bin/env bash
# Pack sail-env and sail-rl (tracked files + git history, at their current commits) for a GPU pod, without
# pushing anything. Extra paths (e.g. a run to warm-start from) are added as they are.
#
#   bash experiments/runpod/pack.sh [runs/train/<run> ...]      (from sail-rl, sail-env cloned next to it)
#   -> runs/runpod/bundle.tar.gz   (copy it to the pod with scp, see README.md)
#
# Uncommitted changes are NOT packed (git ls-files lists the tracked files, but their contents are taken
# from the working tree; commit first so the pod runs a commit you can name).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"   # sail-rl
root="$(dirname "$here")"                     # the folder holding both repos
out="$here/runs/runpod"
mkdir -p "$out"
list="$out/files.txt"
: > "$list"
for repo in sail-env sail-rl; do
    if [ -n "$(git -C "$root/$repo" status --porcelain --untracked-files=no)" ]; then
        echo "warning: $repo has uncommitted changes (packed as they are)" >&2
    fi
    git -C "$root/$repo" ls-files | sed "s|^|$repo/|" >> "$list"
    echo "$repo/.git" >> "$list"
    echo "$repo: $(git -C "$root/$repo" rev-parse --abbrev-ref HEAD) $(git -C "$root/$repo" rev-parse --short HEAD)"
done
for extra in "$@"; do
    echo "sail-rl/${extra#./}" >> "$list"
done
tar -czf "$out/bundle.tar.gz" -C "$root" -T "$list"
echo "-> $out/bundle.tar.gz ($(du -h "$out/bundle.tar.gz" | cut -f1))"
