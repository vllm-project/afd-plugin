#!/bin/bash
#SBATCH --partition=h20
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=afd-async-cross
#SBATCH --output=async-cross-%j.log
set -euo pipefail

# Submit from the checkout with the CUDA/NVSHMEM extension already built.
# GPU GRES on this cluster is untyped; the h20 partition selects H20 nodes.
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-ibrc}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-1G}"
AFD_TEST_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
AFD_TEST_PYTHON="${AFD_TEST_PYTHON:-$PWD/.venv/bin/python}"
AFD_TEST_LOGS="${AFD_TEST_LOGS:-$PWD/.venv/validation-cross-node/$SLURM_JOB_ID}"
mkdir -p "$AFD_TEST_LOGS"

run_oracle() {
    local mode=$1 ranks=$2 port=$3
    srun --nodes=2 --ntasks="$((ranks * 2))" --ntasks-per-node="$ranks" \
        --gres=gpu:2 --gpu-bind=none --kill-on-bad-exit=1 --wait=15 \
        --time=00:10:00 "$AFD_TEST_PYTHON" -m tests.gpu.run_async_gpu_multinode \
        --host "$AFD_TEST_HOST" --port "$port" \
        --attention-ranks="$ranks" --ffn-ranks="$ranks" --mode="$mode" \
        --replays=64 > "$AFD_TEST_LOGS/$mode-${ranks}a${ranks}f.log" 2>&1
}

run_oracle eager 1 6281
run_oracle graph 1 6282
run_oracle precision 1 6283
run_oracle cross_rank_precision 2 6284
run_oracle independent 2 6285
# Hybrid placement: A0/F0 share node 0; A1/F1 share node 1.
srun --nodes=2 --ntasks=4 --ntasks-per-node=2 --distribution=block:block \
    --gres=gpu:2 --gpu-bind=none --kill-on-bad-exit=1 --wait=15 \
    --time=00:10:00 "$AFD_TEST_PYTHON" -m tests.gpu.run_async_gpu_multinode \
    --host "$AFD_TEST_HOST" --port 6286 --attention-ranks=2 --ffn-ranks=2 \
    --interleave-roles --mode=graph --replays=64 \
    > "$AFD_TEST_LOGS/graph-hybrid-2a2f.log" 2>&1
echo "All two-node transport oracles passed; logs: $AFD_TEST_LOGS"
