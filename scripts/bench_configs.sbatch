#!/bin/bash
# Benchmark per-update (per-iteration) wall-clock across hardware configs on the CURRENT
# checked-out code (no code A/B -- this just measures `main` as-is).
#
# Primary question this settles: do multi-GPU workers STARVE at a low --cpus-per-task?
# (compare g4_cpu8 / g4_cpu16 / g4_cpu32 below). Also: single-GPU CPU sweep, num_envs
# saturation, and GPU scaling at a fixed global batch.
#
# Fairness: dedicated Slurm GPUs, identical iters/seed, median of steady-state iters
# (warmup dropped). The warp kernel cache is warmed once so each config skips the recompile.
#
# Submit from the repo root:  sbatch scripts/bench_configs.sbatch
# Override knobs:             ITERS=30 TASK=Mjlab-Skater-Flat-Agibot-X2 sbatch scripts/bench_configs.sbatch
# Output: bench-<jobid>.csv  (+ a table at the end of husky-bench-<jobid>.out)
#
#SBATCH --job-name=husky-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:4      # EDIT: >=4 of ONE type (needed for the 4-GPU rows; single type keeps the warp cache valid)
#SBATCH --cpus-per-task=32            # reserve the MAX any row uses (32); the cpu rows taskset DOWN from here
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=husky-bench-%j.out
##SBATCH --exclusive                  # uncomment for the cleanest CPU numbers (no co-tenant CPU noise)

set -uo pipefail                       # NOT -e: one failing config must not abort the sweep
cd "$SLURM_SUBMIT_DIR"

TASK="${TASK:-Mjlab-Skater-Flat-Unitree-G1}"
ITERS="${ITERS:-25}"
WARMUP="${WARMUP:-8}"
SEED="${SEED:-1}"
NCPU="${SLURM_CPUS_PER_TASK:-$(nproc)}"
RESULTS="bench-${SLURM_JOB_ID:-local}.csv"
export WANDB_MODE=disabled
export MUJOCO_GL=egl

# CPU pinning: Slurm pins the job to specific (non-0-based) core IDs, so absolute
# `taskset -c 0-N` fails. Select the first N cores FROM the allocation; the taskset on the
# launcher is INHERITED by the torchrunx workers, so a 4-GPU run pinned to 8 cores really
# does make its 4 workers share 8 cores -- which is exactly the starvation test.
NOPIN=0
if ! taskset -c "$(python3 -c 'import os;print(sorted(os.sched_getaffinity(0))[0])')" true 2>/dev/null; then
  echo "[WARN] CPU pinning unavailable; cpu-pinned rows will vary thread counts (OMP/MKL) only."; NOPIN=1
fi
pin_cmd() {  # echo 'taskset -c <ids>' for the first $1 allocated cores (nothing if NOPIN)
  [ "$NOPIN" = 1 ] && return
  echo "taskset -c $(python3 -c "import os;c=sorted(os.sched_getaffinity(0));print(','.join(map(str,c[:$1])))")"
}

# "label|cpus|gpus|train-args".  cpus="all" => full allocation (no taskset).
CONFIGS=(
  # single-GPU CPU sweep (expect FLAT past ~4-8 -> a lone env.step process is not core-bound)
  "g1_cpu4|4|1|--env.scene.num-envs 4096"
  "g1_cpu8|8|1|--env.scene.num-envs 4096"
  "g1_cpu32|32|1|--env.scene.num-envs 4096"
  # num_envs saturation (single GPU)
  "g1_e2048|all|1|--env.scene.num-envs 2048"
  "g1_e8192|all|1|--env.scene.num-envs 8192"
  # GPU scaling at a FIXED 4096 global batch (collection should drop ~1.5x @2 GPU)
  "g2_glob4096|all|2|--global-num-envs 4096"
  # *** the multi-GPU CPU-starvation test: 4 GPUs (1024 envs each) sharing 8 / 16 / 32 cores ***
  "g4_cpu8|8|4|--global-num-envs 4096"
  "g4_cpu16|16|4|--global-num-envs 4096"
  "g4_cpu32|32|4|--global-num-envs 4096"
)

# ---------------- run one config; append steady-state median to the CSV ----------------
run_cfg() {  # args: label cpus gpus train-args...
  local label="$1" cpus="$2" gpus="$3"; shift 3
  local log; log="$(mktemp)"
  local thr="$NCPU"; local pin=()
  if [ "$cpus" != "all" ]; then thr="$cpus"; read -r -a pin <<< "$(pin_cmd "$cpus")"; fi
  local ids; ids="$(seq -s ' ' 0 $((gpus-1)))"
  echo ">>> $(date +%T) $label  (cpus=$cpus gpus=$gpus :: $*)"
  "${pin[@]}" env OMP_NUM_THREADS="$thr" MKL_NUM_THREADS="$thr" OPENBLAS_NUM_THREADS="$thr" NUMEXPR_NUM_THREADS="$thr" \
    uv run train "$TASK" --agent.max-iterations "$ITERS" --agent.seed "$SEED" --gpu-ids $ids "$@" \
    > "$log" 2>&1
  if [ $? -ne 0 ]; then echo "    FAILED -- last lines:"; tail -6 "$log"; echo "$label,$cpus,$gpus,NA,NA,NA,0" >> "$RESULTS"; return; fi
  python3 - "$log" "$WARMUP" "$label" "$cpus" "$gpus" >> "$RESULTS" <<'PY'
import sys, re, statistics as st
log, warmup, label, cpus, gpus = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
col, lrn = [], []
for ln in open(log):
    m = re.search(r"collection:\s*([0-9.]+)s,\s*learning\s*([0-9.]+)s", ln)
    if m: col.append(float(m.group(1))); lrn.append(float(m.group(2)))
col, lrn = col[warmup:], lrn[warmup:]
if not col:
    print(f"{label},{cpus},{gpus},NA,NA,NA,0"); raise SystemExit
tot = [c + l for c, l in zip(col, lrn)]
print(f"{label},{cpus},{gpus},{st.median(col):.3f},{st.median(lrn):.3f},{st.median(tot):.3f},{len(col)}")
PY
  echo "    ok: $(tail -1 "$RESULTS")"
}

# ---------------- warm the warp kernel cache ONCE (the expensive startup) ----------------
echo "[$(date +%T)] node=$(hostname)  gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)  cpus/task=$NCPU  pinning=$([ $NOPIN = 0 ] && echo on || echo off)"
echo "[$(date +%T)] warming warp kernel cache (~/.cache/warp) -- one-time compile..."
_wlog="$(mktemp)"
uv run train "$TASK" --agent.max-iterations 2 --env.scene.num-envs 1024 --gpu-ids 0 > "$_wlog" 2>&1 \
  && echo "[$(date +%T)] cache warm -- subsequent runs skip kernel compilation." \
  || { echo "[$(date +%T)] WARNING: warmup run errored (real train/env issue, not taskset). Last lines:"; tail -10 "$_wlog"; }

# ---------------- the sweep ----------------
echo "config,cpus,gpus,median_collection_s,median_learning_s,median_total_s,n" > "$RESULTS"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label cpus gpus args <<< "$cfg"
  run_cfg "$label" "$cpus" "$gpus" $args
done

echo; echo "=============== RESULTS ($RESULTS) ==============="
column -s, -t "$RESULTS"
echo
echo "Read it as:"
echo " * MULTI-GPU CPU TEST: g4_cpu8 vs g4_cpu16 vs g4_cpu32 -- if median_collection_s DROPS as cores"
echo "   rise, the 4 workers were CPU-starved at the low counts -> use ~8 cpus/GPU. If flat, 8 total is enough."
echo " * single-GPU CPU sweep: g1_cpu4/8/32 -- expected flat (a lone env.step process is not core-bound)."
echo " * GPU scaling: g1_cpu32 (1 GPU) vs g2_glob4096 (2 GPU) vs g4_cpu32 (4 GPU) -- same 4096 global batch."
echo " * num_envs: g1_e2048 / *_e4096 / g1_e8192 -- where per-update time saturates."
