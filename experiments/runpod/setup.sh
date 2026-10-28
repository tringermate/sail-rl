#!/usr/bin/env bash
# On the GPU pod: unpack the bundle made by pack.sh into /workspace and install sail_env with CUDA JAX.
#
#   (bundle.tar.gz copied to /workspace with scp, see README.md)
#   cd /workspace
#   tar --no-same-owner -xzf bundle.tar.gz sail-rl/experiments/runpod/setup.sh && bash sail-rl/experiments/runpod/setup.sh
#
# Then:  source /workspace/venv/bin/activate && cd /workspace/sail-rl
#        export WANDB_API_KEY=...   (only if the configs log to W&B; never commit it)
set -euo pipefail
cd /workspace
tar --no-same-owner -xzf bundle.tar.gz
command -v uv >/dev/null || pip install -q uv
[ -d venv ] || uv venv -q -p 3.12 venv
source venv/bin/activate
uv pip install -q -e ./sail-env "jax[cuda12]" "wandb>=0.18" pytest
python - <<'EOF'
import jax
print("jax", jax.__version__, "devices:", jax.devices())
assert jax.default_backend() == "gpu", "JAX does not see the GPU"
EOF
for repo in sail-env sail-rl; do echo "$repo: $(git -C $repo rev-parse --abbrev-ref HEAD) $(git -C $repo rev-parse --short HEAD)"; done
echo "ready: source /workspace/venv/bin/activate && cd /workspace/sail-rl"
