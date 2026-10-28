# Training on a Runpod GPU pod

The training is JAX end to end (environment, vectorised boats and PPO in one compiled loop), so a GPU runs
thousands of races in parallel. The speed comes from raising `ppo.num_envs`; at 64 envs a GPU is mostly idle.
More envs per update changes the training (bigger batches), so lr / minibatches / steps need retuning: compare
against a known CPU result before trusting a GPU run.

## 1. Pack the code (on the laptop, nothing is pushed)

    bash experiments/runpod/pack.sh                         # code only
    bash experiments/runpod/pack.sh runs/train/<run>        # + a run to warm-start from (init_from)
    scp -P <port> -i ~/.ssh/id_ed25519_runpod runs/runpod/bundle.tar.gz root@<ip>:/workspace/

(`runpodctl` is blocked on this laptop by Windows App Control; Windows' own OpenSSH works. The pod needs a
public IP with SSH over exposed TCP; its address and port are on the pod's Connect tab / from the MCP tools.)

## 2. Set up the pod

A pod with one GPU (an RTX 4090 / A40 / L40S class card is plenty), any CUDA 12 image (e.g. Runpod PyTorch),
a volume on /workspace (survives a pod restart). Then, on the pod:

    cd /workspace
    tar --no-same-owner -xzf bundle.tar.gz sail-rl/experiments/runpod/setup.sh && bash sail-rl/experiments/runpod/setup.sh
    source /workspace/venv/bin/activate && cd /workspace/sail-rl

W&B: `export WANDB_API_KEY=...` in the pod's shell (or as a pod environment variable), never in the repo.

## 3. Benchmark first

    python experiments/bench_throughput.py experiments/configs/ladder/s4k_full_kick.yaml --envs 64 512 2048 4096

Run the same on the laptop for the CPU numbers (runs/bench_throughput.csv collects both).

## 4. Train in the background

    bash experiments/runpod/run_queue.sh experiments/configs/<...>.yaml[:init] ...   # loads WANDB_* from the pod env, checks the GPU
    tail -f runs/queue.log

Results: W&B live; the run folders come back with
`scp -r -P <port> -i ~/.ssh/id_ed25519_runpod root@<ip>:/workspace/sail-rl/runs/train/<run> runs/train/`.

**Stop the pod when the queue is done**: a running pod is billed by the second, a stopped one only for its volume.
