#!/bin/bash
# Find the IDEAL per-GPU packing for a hyperparameter sweep: how many training runs to
# co-locate on ONE GPU, and at what --env.scene.num-envs each, to MAXIMIZE aggregate
# throughput (sum of the co-located runs' env-steps/s). This workload is latency/sync-bound,
# so a single run leaves the GPU's compute mostly idle -- packing several runs fills it.
# The cap is GPU MEMORY and CPU (each run has its own launch loop), not GPU compute. We also
# measure MPS off vs on (MPS lets the runs share SMs concurrently instead of time-slicing).
#
# Whatever packing wins here, replicate it on every GPU you have for the sweep.
#
# Run ONCE:   sbatch scripts/bench_sweep_colocation.sbatch
# Knobs:      TASK=Mjlab-Skater-Flat-Unitree-G1 ITERS=30 NO_MPS=1 sbatch scripts/bench_sweep_colocation.sbatch
# Output: sweep-bench-<jobid>.csv + a ranked table and a recommendation in husky-sweepbench-<jobid>.out
#
#SBATCH --job-name=husky-sweepbench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40s:1     # ONE GPU -- we measure per-GPU packing. EDIT the type (h100/nvidia_l40s/...).
#SBATCH --cpus-per-task=32           # generous: every co-located run needs CPU for its launch loop
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=husky-sweepbench-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

TASK="${TASK:-Mjlab-Skater-Flat-Agibot-X2}"
ITERS="${ITERS:-26}"
WARMUP="${WARMUP:-10}"                 # drop per run: build-settle + concurrent-startup transient
SEED="${SEED:-1}"
GPU=0
RESULTS="sweep-bench-${SLURM_JOB_ID:-local}.csv"
SCR="${SLURM_TMPDIR:-/tmp}/sweepbench.${SLURM_JOB_ID:-$$}"; mkdir -p "$SCR"
export WANDB_MODE=disabled MUJOCO_GL=egl NCCL_P2P_DISABLE=1

# "(N concurrent runs) | (per-run num_envs)". Higher N needs smaller envs to fit memory.
# Each E has an N=1 row so we can report per-run slowdown vs running solo.
GRID=(
  "1|4096" "2|4096"
  "1|2048" "2|2048" "4|2048"
  "1|1024" "2|1024" "4|1024" "8|1024"
)

# ---------- MPS control (concurrent SM sharing; usually the big co-location lever) ----------
start_mps() {
  [ "${NO_MPS:-0}" = 1 ] && return 1
  export CUDA_MPS_PIPE_DIRECTORY="$SCR/mps_pipe" CUDA_MPS_LOG_DIRECTORY="$SCR/mps_log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d 2>/dev/null && echo "get_server_list" | nvidia-cuda-mps-control >/dev/null 2>&1
}
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null' EXIT

# ---------- run one (mps, N, E) config: launch N concurrent runs, measure aggregate ----------
run_config() {  # mps N E
  local mps="$1" N="$2" E="$3"
  local logs=() pids=()
  echo ">>> $(date +%T)  mps=$mps  N=$N  E=$E  (launch $N concurrent runs @ $E envs on gpu $GPU)"
  local memlog="$SCR/mem_${mps}_${N}_${E}.txt"; : > "$memlog"
  # peak memory on the busiest GPU (= the one we pack); robust to Slurm/CUDA index remapping.
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; sleep 3; done ) > "$memlog" &
  local mempid=$!
  for i in $(seq 0 $((N-1))); do
    local log="$SCR/run_${mps}_${N}_${E}_${i}.log"; logs+=("$log")
    # Inherit Slurm's CUDA_VISIBLE_DEVICES (the allocated GPU); --gpu-ids 0 picks it. Do NOT
    # override CVD to a fixed index -- Slurm may have allocated a different physical GPU.
    uv run train "$TASK" --gpu-ids 0 \
      --env.scene.num-envs "$E" --agent.seed $((SEED+i)) \
      --agent.run-name "swp_${mps}_N${N}_E${E}_i${i}" --agent.max-iterations "$ITERS" \
      > "$log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  kill "$mempid" 2>/dev/null; wait "$mempid" 2>/dev/null
  local peak; peak=$(sort -n "$memlog" 2>/dev/null | tail -1); peak=${peak:-0}
  python3 - "$mps" "$N" "$E" "$WARMUP" "$peak" "$fail" "${logs[@]}" >> "$RESULTS" <<'PY'
import sys, re, statistics as st
mps, N, E, warmup, peak, fail = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
per = []
for lg in sys.argv[7:]:
    vals = []
    try:
        for ln in open(lg):
            m = re.search(r"Computation:\s*([0-9.]+)\s*steps/s", ln)
            if m: vals.append(float(m.group(1)))
    except FileNotFoundError:
        pass
    vals = vals[warmup:]
    if vals: per.append(st.median(vals))
status = "ok" if (len(per) == N and fail == "0") else ("OOM/FAIL" if fail != "0" else f"partial {len(per)}/{N}")
agg = round(sum(per)) if per else 0
permed = round(st.median(per)) if per else 0
print(f"{mps},{N},{E},{agg},{permed},,{peak},{status}")
PY
  echo "    $(tail -1 "$RESULTS")"
}

run_grid() {  # mps_label
  for cfg in "${GRID[@]}"; do IFS='|' read -r N E <<< "$cfg"; run_config "$1" "$N" "$E"; done
}

# ---------- warm the warp kernel cache once ----------
echo "[$(date +%T)] node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)  cpus=${SLURM_CPUS_PER_TASK:-?}  (tip: --exclusive gives the cleanest memory numbers)"
echo "[$(date +%T)] warming warp cache..."
uv run train "$TASK" --agent.max-iterations 2 --env.scene.num-envs 1024 --gpu-ids 0 >/dev/null 2>&1 \
  && echo "[$(date +%T)] cache warm." || echo "[$(date +%T)] warmup errored (continuing)."

echo "mps,n_runs,num_envs,agg_steps_per_s,per_run_steps_per_s,per_run_vs_solo,peak_gpu_mem_MB,status" > "$RESULTS"

# Pass 1: time-sliced (no MPS).  Pass 2: MPS (concurrent SM sharing), if it starts.
run_grid "off"
if start_mps; then echo "[INFO] MPS started -- second pass shares SMs concurrently."; run_grid "on"; stop_mps
else echo "[INFO] MPS unavailable/disabled -- skipping the MPS pass (time-sliced numbers only)."; fi

# ---------- fill per-run-vs-solo, print the ranked table + recommendation ----------
python3 - "$RESULTS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    raise SystemExit("[!] no results rows")
solo = {(r['mps'], r['num_envs']): float(r['per_run_steps_per_s'])
        for r in rows if r['n_runs'] == '1' and r['status'] == 'ok' and r['per_run_steps_per_s'] not in ('', '0')}
for r in rows:
    s = solo.get((r['mps'], r['num_envs'])); pr = float(r['per_run_steps_per_s'] or 0)
    r['per_run_vs_solo'] = f"{pr/s:.2f}" if (s and pr) else ""
w = csv.DictWriter(open(sys.argv[1], 'w'), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

H = ["mps","n_runs","num_envs","agg_steps_per_s","per_run_steps_per_s","per_run_vs_solo","peak_gpu_mem_MB","status"]
print("\n=================================== RESULTS ===================================")
print("  ".join(f"{h:>15}" for h in H))
for r in sorted(rows, key=lambda r: (r['mps'], int(r['num_envs']), int(r['n_runs']))):
    print("  ".join(f"{r[h]:>15}" for h in H))

ok = [r for r in rows if r['status'] == 'ok' and float(r['agg_steps_per_s']) > 0]
if ok:
    best = max(ok, key=lambda r: float(r['agg_steps_per_s']))
    solos = [r for r in ok if r['n_runs'] == '1']
    base = max(solos, key=lambda r: float(r['agg_steps_per_s'])) if solos else None
    print("\n----------------------------------- RECOMMENDATION -----------------------------------")
    print(f"  Pack {best['n_runs']} runs @ {best['num_envs']} envs each per GPU (MPS {best['mps']}):")
    print(f"    aggregate {best['agg_steps_per_s']} env-steps/s/GPU  |  each run at {best['per_run_vs_solo']}x its solo speed  |  peak mem {best['peak_gpu_mem_MB']} MB")
    if base:
        print(f"    => {float(best['agg_steps_per_s'])/float(base['agg_steps_per_s']):.2f}x the throughput of 1 run/GPU. Replicate this packing on every GPU.")
    # show the best at each num_envs so the user can trade throughput for batch size
    print("\n  best aggregate at each num_envs (bigger envs = bigger PPO batch = better sample efficiency):")
    for E in sorted({r['num_envs'] for r in ok}, key=int):
        b = max((r for r in ok if r['num_envs'] == E), key=lambda r: float(r['agg_steps_per_s']))
        print(f"    {E:>5} envs -> {b['n_runs']} runs/GPU, {b['agg_steps_per_s']} steps/s/GPU, MPS {b['mps']}, mem {b['peak_gpu_mem_MB']} MB")
    print("\n  NOTE: smaller num_envs gives more aggregate throughput but a smaller batch per run")
    print("        (more iterations to converge). Prefer the largest num_envs whose aggregate is")
    print("        within ~10-15% of the global max, unless your sweep tolerates small batches.")
else:
    print("\n[!] no config succeeded -- check the .out for OOM/errors (reduce GRID num_envs).")
PY

# tidy the benchmark's throwaway run dirs (gitignored logs/), keep nothing
rm -rf logs/rsl_rl/*/*_swp_* 2>/dev/null
echo; echo "results -> $RESULTS"
