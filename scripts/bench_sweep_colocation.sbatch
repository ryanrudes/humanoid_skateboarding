#!/bin/bash
# Find the IDEAL per-GPU packing for a hyperparameter sweep, on ANY GPU, with no manual
# tweaking. It auto-discovers each GPU's limits: for each num_envs it ramps the co-location
# density (1, 2, 4, 8, ...) until the runs OOM or aggregate throughput plateaus -- so a 16 GB
# P100 stops early and a 288 GB B300 ramps deep, automatically. MPS off vs on is measured too
# (MPS is what makes co-location pay off; without it the GPU time-slices and aggregate is flat).
#
# The ONLY per-GPU input is the GPU you allocate -- pass it to sbatch (no file edits):
#   sbatch --gres=gpu:nvidia_l40s:1 scripts/bench_sweep_colocation.sbatch
#   sbatch --gres=gpu:p100:1        scripts/bench_sweep_colocation.sbatch
#   sbatch --gres=gpu:h100:1        scripts/bench_sweep_colocation.sbatch
#   sbatch --gres=gpu:nvidia_h200:1 scripts/bench_sweep_colocation.sbatch
# (Fat-CPU nodes help co-location -- add e.g. --cpus-per-task=64 --mem=0 on the sbatch line.)
#
# Knobs (env):  TASK=... ITERS=30 NUM_ENVS_SET="2048 4096" MAX_N=64 NO_MPS=1 sbatch ...
# Output: sweep-bench-<jobid>.csv + a ranked table and recommendation in husky-sweepbench-<jobid>.out
#
#SBATCH --job-name=husky-sweepbench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # DEFAULT (overridden by the sbatch --gres above; typed-gres clusters require the override)
#SBATCH --cpus-per-task=32           # bump on fat nodes -- co-location wants ~4-8 cores/run
#SBATCH --mem=128G
#SBATCH --time=08:00:00              # generous: big GPUs ramp deep. Trim NUM_ENVS_SET / MAX_N to shorten.
#SBATCH --output=husky-sweepbench-%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

TASK="${TASK:-Mjlab-Skater-Flat-Agibot-X2}"
ITERS="${ITERS:-24}"
WARMUP="${WARMUP:-10}"                          # drop per run: build-settle + concurrent-startup transient
SEED="${SEED:-1}"
NUM_ENVS_SET="${NUM_ENVS_SET:-1024 2048 4096 8192}"
MAX_N="${MAX_N:-32}"                            # hard cap on co-location density (backstop; the plateau usually stops first)
PLATEAU="${PLATEAU:-1.05}"                      # stop ramping when doubling N gives < this x more aggregate throughput
RESULTS="sweep-bench-${SLURM_JOB_ID:-local}.csv"
SCR="${SLURM_TMPDIR:-/tmp}/sweepbench.${SLURM_JOB_ID:-$$}"; mkdir -p "$SCR"
export WANDB_MODE=disabled MUJOCO_GL=egl NCCL_P2P_DISABLE=1

# ---------------- MPS control (concurrent SM sharing; the co-location lever) ----------------
start_mps() {
  [ "${NO_MPS:-0}" = 1 ] && return 1
  export CUDA_MPS_PIPE_DIRECTORY="$SCR/mps_pipe" CUDA_MPS_LOG_DIRECTORY="$SCR/mps_log"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d 2>/dev/null && echo "get_server_list" | nvidia-cuda-mps-control >/dev/null 2>&1
}
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null' EXIT

# ---------------- run ONE (mps,N,E) config: N concurrent runs, set LAST_AGG/LAST_STATUS ----------------
run_config() {
  local mps="$1" N="$2" E="$3"
  local logs=() pids=()
  local memlog="$SCR/mem_${mps}_${N}_${E}.txt"; : > "$memlog"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; sleep 3; done ) > "$memlog" &
  local mempid=$!
  for i in $(seq 0 $((N-1))); do
    local log="$SCR/run_${mps}_${N}_${E}_${i}.log"; logs+=("$log")
    # Inherit Slurm's CUDA_VISIBLE_DEVICES (the allocated GPU); --gpu-ids 0 picks it.
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
  local row
  row=$(python3 - "$mps" "$N" "$E" "$WARMUP" "$peak" "$fail" "${logs[@]}" <<'PY'
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
)
  echo "$row" >> "$RESULTS"
  LAST_AGG=$(printf '%s' "$row" | cut -d, -f4)
  LAST_STATUS=$(printf '%s' "$row" | cut -d, -f8)
  echo "    mps=$mps N=$N E=$E -> agg=${LAST_AGG} steps/s  mem=${peak}MB  [$LAST_STATUS]"
}

# ---------------- ramp density for one (mps, num_envs) until OOM or throughput plateau ----------------
ramp() {
  local mps="$1" E="$2" prev=0 N=1
  echo ">>> $(date +%T)  ramp mps=$mps num_envs=$E"
  while [ "$N" -le "$MAX_N" ]; do
    run_config "$mps" "$N" "$E"
    if [ "$LAST_STATUS" != "ok" ]; then
      echo "    stop: N=$N did not run cleanly ($LAST_STATUS) -- past this GPU's limit for $E envs."; break
    fi
    if [ "$N" -gt 1 ] && awk "BEGIN{exit !($LAST_AGG < $prev * $PLATEAU)}"; then
      echo "    stop: throughput plateau at N=$N (agg $LAST_AGG <= ${PLATEAU}x prev $prev)."; break
    fi
    prev="$LAST_AGG"; N=$((N * 2))
  done
}

# ---------------- warm the warp kernel cache once ----------------
echo "[$(date +%T)] node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "[$(date +%T)] warming warp cache..."
uv run train "$TASK" --agent.max-iterations 2 --env.scene.num-envs 1024 --gpu-ids 0 >/dev/null 2>&1 \
  && echo "[$(date +%T)] cache warm." || echo "[$(date +%T)] warmup errored (continuing; a real run will compile once)."

echo "mps,n_runs,num_envs,agg_steps_per_s,per_run_steps_per_s,per_run_vs_solo,peak_gpu_mem_MB,status" > "$RESULTS"

# Pass 1: time-sliced (no MPS) -- ramps short (aggregate is flat without MPS).
for E in $NUM_ENVS_SET; do ramp "off" "$E"; done
# Pass 2: MPS (concurrent SM sharing) -- ramps until the GPU's real ceiling.
if start_mps; then echo "[INFO] MPS started."; for E in $NUM_ENVS_SET; do ramp "on" "$E"; done; stop_mps
else echo "[INFO] MPS unavailable/disabled -- time-sliced numbers only."; fi

# ---------------- fill per-run-vs-solo, ranked table + recommendation ----------------
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
    base = max((r for r in ok if r['n_runs'] == '1'), key=lambda r: float(r['agg_steps_per_s']), default=None)
    print("\n----------------------------------- RECOMMENDATION -----------------------------------")
    print(f"  Pack {best['n_runs']} runs @ {best['num_envs']} envs each per GPU (MPS {best['mps']}):")
    print(f"    aggregate {best['agg_steps_per_s']} env-steps/s/GPU  |  each run at {best['per_run_vs_solo']}x its solo speed  |  peak mem {best['peak_gpu_mem_MB']} MB")
    if base:
        print(f"    => {float(best['agg_steps_per_s'])/float(base['agg_steps_per_s']):.2f}x the throughput of 1 run/GPU. Replicate this packing on every GPU.")
    print("\n  best aggregate at each num_envs (bigger envs = bigger PPO batch = better sample efficiency):")
    for E in sorted({r['num_envs'] for r in ok}, key=int):
        b = max((r for r in ok if r['num_envs'] == E), key=lambda r: float(r['agg_steps_per_s']))
        print(f"    {E:>5} envs -> {b['n_runs']} runs/GPU, {b['agg_steps_per_s']} steps/s/GPU, MPS {b['mps']}, mem {b['peak_gpu_mem_MB']} MB")
    print("\n  NOTE: the launcher default is the largest num_envs whose aggregate is near the max")
    print("        (faithful batch); drop to a smaller num_envs only if your sweep tolerates it.")
else:
    print("\n[!] no config ran cleanly -- check the .out (e.g. the GPU/driver may need a newer warp/CUDA, or num_envs too large).")
PY

rm -rf logs/rsl_rl/*/*_swp_* 2>/dev/null      # tidy the benchmark's throwaway run dirs (gitignored)
echo; echo "results -> $RESULTS"
