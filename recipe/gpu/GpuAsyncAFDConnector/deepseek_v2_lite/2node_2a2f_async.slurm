#!/bin/bash
# Two-node placement: node 0 hosts 2 Attention GPUs; node 1 hosts 2 FFN GPUs.
#SBATCH --partition=h20
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=afd-async-v2
#SBATCH --output=afd-async-v2-%j.log
set -euo pipefail

# Requires the pinned CUDA runtime and AFD_BUILD_GPU_ASYNC_OPS=1 installation
# on both nodes. Model files must be available at these paths on each node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-ibrc}"
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-1G}"
export VLLM_PLUGINS=afd VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BIN="${VLLM_BIN:-$SLURM_SUBMIT_DIR/.venv/bin/vllm}"
export PATH="$(dirname "$VLLM_BIN"):$PATH"

if [[ "${1:-}" != --role ]]; then
    cd "$SLURM_SUBMIT_DIR"
    export AFD_HOST="${AFD_HOST:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)}"
    export AFD_PORT="${AFD_PORT:-6281}"
    AFD_RECIPE="$SLURM_SUBMIT_DIR/recipe/gpu/GpuAsyncAFDConnector/deepseek_v2_lite/2node_2a2f_async.slurm"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:2 \
        --kill-on-bad-exit=1 bash "$AFD_RECIPE" --role "$@"
    exit
fi
shift

if [[ "$SLURM_PROCID" == 0 ]]; then
    AFD_ROLE=attention
    AFD_MODEL="${ATTENTION_MODEL_PATH:-${MODEL_PATH:?Set MODEL_PATH or both role model paths}}"
    AFD_API_PORT="${ATTENTION_API_PORT:-8000}"
else
    AFD_ROLE=ffn
    AFD_MODEL="${FFN_MODEL_PATH:-${MODEL_PATH:?Set MODEL_PATH or both role model paths}}"
    AFD_API_PORT="${FFN_API_PORT:-8001}"
fi

exec "$VLLM_BIN" serve "$AFD_MODEL" \
    --host 0.0.0.0 --port "$AFD_API_PORT" --served-model-name deepseek-v2-lite \
    --trust-remote-code --dtype bfloat16 --seed 0 \
    --data-parallel-size 2 --tensor-parallel-size 1 \
    --enable-expert-parallel --enable-ep-weight-filter \
    --api-server-count 1 --shutdown-timeout 120 \
    --max-model-len 4096 --max-num-batched-tokens 1024 --max-num-seqs 8 \
    --gpu-memory-utilization 0.5 \
    --enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --cudagraph-capture-sizes 2 4 8 \
    --additional-config "{\"afd\":{\"role\":\"$AFD_ROLE\",\"connector\":\"GpuAsyncAFDConnector\",\"async\":true,\"compute_gate_on_attention\":true,\"num_attention_ranks\":2,\"num_ffn_ranks\":2,\"host\":\"$AFD_HOST\",\"port\":$AFD_PORT}}" \
    "$@"
