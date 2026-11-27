#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Train the 30 Hz Human24 FSQVAE on one or more GPUs with 128 windows per GPU (8 GPUs per node by default).

  scripts/train_human24_paper.sh

Set DATA_DIR to the original shared Human24 NPZ directory (containing skeleton.json).
OUTPUT_DIR defaults to outputs/human24_fsq_paper_recheck under the repo. Each GPU reads NPZ
windows directly. DATA_WORKERS controls reader threads per GPU (default 4).
Set PYTHON_BIN to a Python executable with PyTorch, NumPy and SciPy installed.
This launcher trains the original scalar FSQ. STOP_AFTER optionally saves
and validates at an absolute step. INIT_FROM optionally starts a new run at the
step recorded in an existing Human24 checkpoint. SKATING_WEIGHT defaults to 0.01;
set it to 0 to disable the training term.

Run once per node. PAI DLC supplies WORLD_SIZE (node count), RANK (node index),
NPROC_PER_NODE, MASTER_ADDR and MASTER_PORT. Else set NNODES and NODE_RANK
explicitly. Under torchrun an existing LOCAL_RANK uses the existing group.
Run the same command after preemption to resume the latest saved step.
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${DATA_DIR:-/mnt/data/datas/bones-seed/human24}
output_dir=${OUTPUT_DIR:-"$repo_dir/outputs/human24_fsq_paper_recheck"}
python_bin=${PYTHON_BIN:-python}
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [[ ! -f "$data_dir/skeleton.json" ]]; then
  echo "Expected Human24 skeleton.json under DATA_DIR=$data_dir" >&2
  exit 1
fi
mkdir -p -- "$output_dir"
args=(-m ardy.human24.paper_train --data "$data_dir" --workers "${DATA_WORKERS:-4}" --output "$output_dir" --auto-resume)
if [[ -n ${SKATING_WEIGHT:-} ]]; then args+=(--skating-weight "$SKATING_WEIGHT"); fi
if [[ -n ${STOP_AFTER:-} ]]; then args+=(--stop-after "$STOP_AFTER"); fi
if [[ -n ${INIT_FROM:-} ]]; then args+=(--init-from "$INIT_FROM"); fi
if [[ -n "${LOCAL_RANK:-}" ]]; then
  # An external torchrun already launched one worker per GPU.
  exec "$python_bin" -u "${args[@]}"
fi

nodes=${NNODES:-${WORLD_SIZE:-1}}
procs=${NPROC_PER_NODE:-8}
if (( nodes < 1 || procs < 1 )); then
  echo "Invalid topology: nodes=$nodes processes_per_node=$procs" >&2
  exit 2
fi
if (( nodes == 1 && procs == 1 )); then
  exec "$python_bin" -u "${args[@]}"
elif (( nodes == 1 )); then
  launch=(--standalone --nnodes=1)
else
  node_rank=${NODE_RANK:-${RANK:?Missing node index: set NODE_RANK or RANK}}
  : "${MASTER_ADDR:?Missing address of the rank-zero node}"
  : "${MASTER_PORT:?Missing rendezvous port}"
  if (( node_rank < 0 || node_rank >= nodes )); then echo "Invalid node rank $node_rank for $nodes nodes" >&2; exit 2; fi
  launch=(--nnodes="$nodes" --node-rank="$node_rank" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT")
fi
echo "Starting torchrun: nodes=$nodes node_rank=${node_rank:-0} gpus_per_node=$procs per_gpu_batch=128 global_batch=$((nodes * procs * 128))"
exec "$python_bin" -u -m torch.distributed.run "${launch[@]}" --nproc-per-node="$procs" "${args[@]}"
