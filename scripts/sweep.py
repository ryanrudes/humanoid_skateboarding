"""Run a hyperparameter sweep by co-locating training runs across GPUs (with MPS).

Why MPS: this workload is latency/sync-bound, so a single run leaves the GPU ~40% idle.
Packing several runs per GPU recovers that -- but ONLY under NVIDIA MPS (without it the GPU
time-slices and aggregate throughput is flat; see scripts/bench_sweep_colocation.sbatch).
On Ada-class GPUs the sweet spot is ~2 runs/GPU at 4096 envs (faithful batch) or ~4/GPU at
2048 (more throughput, smaller batch) -> ~1.5-1.6x the per-GPU sweep rate.

This launcher: starts MPS once for the node, expands a parameter grid (Cartesian over --param,
or a --grid file), and schedules the points across (GPUs x --per-gpu) slots -- each run pinned
to one GPU, named/logged by its hyperparameters, new runs launched as slots free.

Run it INSIDE a GPU allocation (it auto-detects the allocated GPUs from CUDA_VISIBLE_DEVICES).
Example sbatch wrapper:
    #!/bin/bash
    #SBATCH --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:nvidia_l40s:4
    #SBATCH --cpus-per-task=32 --mem=192G --time=24:00:00
    cd "$SLURM_SUBMIT_DIR"
    uv run python scripts/sweep.py --per-gpu 2 --num-envs 4096 \
        --base-args "--agent.max-iterations 50000 --agent.save-interval 500 --agent.wandb-project Skateboarding" \
        --param "agent.algorithm.learning_rate=1e-3,5e-4,2e-4" \
        --param "agent.algorithm.entropy_coef=0.005,0.01"

Quick local check:  uv run python scripts/sweep.py --gpus 0,1 --param "agent.seed=1,2" --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def detect_alloc_gpus() -> list[str]:
  """Physical GPU ids this process may use. Inside Slurm that's the allocation
  (CUDA_VISIBLE_DEVICES); otherwise every GPU nvidia-smi reports."""
  cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
  if cvd:
    return [x for x in cvd.split(",") if x != ""]
  try:
    out = subprocess.check_output(
      ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    return [l.strip() for l in out.splitlines() if l.strip() != ""]
  except Exception:
    return ["0"]


def expand_grid(params: list[tuple[str, list[str]]], grid_file: str | None) -> list[dict]:
  """Cartesian product of --param specs, or raw flag-lines from --grid."""
  if grid_file:
    points = []
    for line in Path(grid_file).read_text().splitlines():
      line = line.strip()
      if line and not line.startswith("#"):
        points.append({"__raw__": line})  # raw extra-flags string
    return points
  if not params:
    return [{}]
  keys = [k for k, _ in params]
  return [dict(zip(keys, combo)) for combo in itertools.product(*[v for _, v in params])]


def point_name(point: dict, idx: int) -> str:
  """Readable, unique run-name from a point's params (idx-prefixed for uniqueness)."""
  if "__raw__" in point:
    return f"swp{idx:03d}"  # raw --grid lines: name by index (use --param for descriptive names)
  body = "_".join(f"{k.split('.')[-1]}{v}" for k, v in point.items())
  body = re.sub(r"[^A-Za-z0-9._=+-]+", "-", body)[:48]
  return f"swp{idx:03d}" + (f"_{body}" if body else "")


def build_cmd(task: str, num_envs: int, base_args: list[str], point: dict, name: str) -> list[str]:
  cmd = ["uv", "run", "train", task, "--gpu-ids", "0",
         "--env.scene.num-envs", str(num_envs), "--agent.run-name", name]
  cmd += base_args
  if "__raw__" in point:
    cmd += shlex.split(point["__raw__"])
  else:
    for k, v in point.items():
      cmd += [f"--{k.replace('_', '-')}", str(v)]
  return cmd


def start_mps(pipe_dir: Path) -> bool:
  pipe_dir.mkdir(parents=True, exist_ok=True)
  (pipe_dir.parent / "mps_log").mkdir(parents=True, exist_ok=True)
  os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
  os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(pipe_dir.parent / "mps_log")
  try:
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n", text=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=True)
    return True
  except Exception:
    os.environ.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    os.environ.pop("CUDA_MPS_LOG_DIRECTORY", None)
    return False


def stop_mps() -> None:
  try:
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
  except Exception:
    pass


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Mjlab-Skater-Flat-Agibot-X2")
  p.add_argument("--gpus", default="all", help="'all' or logical indices into the allocation, e.g. 0,1,2,3")
  p.add_argument("--per-gpu", type=int, default=2, help="runs co-located per GPU (MPS). 2@4096 or 4@2048 on Ada.")
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--base-args", default="", help="flags common to every run, e.g. '--agent.max-iterations 50000 --agent.wandb-project Skateboarding'")
  p.add_argument("--param", action="append", default=[], metavar="path=v1,v2,...",
                 help="config path + comma values to sweep (Cartesian over repeats), e.g. agent.algorithm.learning_rate=1e-3,5e-4")
  p.add_argument("--grid", default=None, help="file of raw extra-flag lines (one run each); alternative to --param")
  p.add_argument("--mps", dest="mps", action="store_true", default=True)
  p.add_argument("--no-mps", dest="mps", action="store_false")
  p.add_argument("--logdir", default=None)
  p.add_argument("--dry-run", action="store_true")
  args = p.parse_args()

  params = []
  for spec in args.param:
    if "=" not in spec:
      sys.exit(f"--param must be path=v1,v2,...: got {spec!r}")
    k, vs = spec.split("=", 1)
    params.append((k.strip(), [v.strip() for v in vs.split(",") if v.strip()]))
  grid = expand_grid(params, args.grid)
  base_args = shlex.split(args.base_args)

  phys = detect_alloc_gpus()
  logical = list(range(len(phys))) if args.gpus == "all" else [int(x) for x in args.gpus.split(",")]
  for g in logical:
    if g >= len(phys):
      sys.exit(f"--gpus references logical GPU {g} but only {len(phys)} are allocated ({phys})")
  slots = [phys[g] for g in logical for _ in range(args.per_gpu)]  # physical GPU per slot

  ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  logdir = Path(args.logdir or f"sweep_logs/{ts}")

  print(f"sweep: {len(grid)} runs  |  {len(logical)} GPU(s) x {args.per_gpu}/GPU = {len(slots)} slots  "
        f"|  task={args.task}  num_envs={args.num_envs}  MPS={'on' if args.mps else 'off'}")
  print(f"  GPUs (physical): {[phys[g] for g in logical]}   logs: {logdir}/")
  if args.per_gpu > 1 and not args.mps:
    print("  [WARN] --per-gpu>1 without MPS: co-located runs will TIME-SLICE -> ~no aggregate gain (see the benchmark).")

  if args.dry_run:
    print("\n--- DRY RUN (commands; not launching) ---")
    for i, pt in enumerate(grid):
      nm = point_name(pt, i)
      print(f"[slot gpu {slots[i % len(slots)]}] {nm}:\n    " + " ".join(shlex.quote(c) for c in build_cmd(args.task, args.num_envs, base_args, pt, nm)))
    return

  logdir.mkdir(parents=True, exist_ok=True)
  mps_on = False
  if args.mps:
    mps_on = start_mps(logdir / "mps_pipe")
    print("  MPS daemon: " + ("started (runs share SMs concurrently)" if mps_on
                              else "[WARN] could not start -- continuing TIME-SLICED (co-location won't help)."))

  running: dict[int, tuple[subprocess.Popen, int, object]] = {}  # slot -> (proc, point_idx, logfile)
  queue = list(range(len(grid)))
  results: dict[int, int] = {}

  def shutdown(*_):
    print("\n[shutdown] terminating running runs...")
    for proc, _, _ in running.values():
      proc.terminate()
    if mps_on:
      stop_mps()
    sys.exit(1)
  signal.signal(signal.SIGINT, shutdown)
  signal.signal(signal.SIGTERM, shutdown)

  try:
    while queue or running:
      for si in range(len(slots)):
        if si in running or not queue:
          continue
        pidx = queue.pop(0)
        nm = point_name(grid[pidx], pidx)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[si])  # pin to this slot's GPU
        lf = open(logdir / f"{nm}.log", "w")
        proc = subprocess.Popen(build_cmd(args.task, args.num_envs, base_args, grid[pidx], nm),
                                env=env, stdout=lf, stderr=subprocess.STDOUT)
        running[si] = (proc, pidx, lf)
        print(f"[{time.strftime('%H:%M:%S')}] launch  gpu {slots[si]}  {nm}   ({len(results)+len(running)}/{len(grid)} active+done)")
      for si, (proc, pidx, lf) in list(running.items()):
        rc = proc.poll()
        if rc is not None:
          lf.close()
          results[pidx] = rc
          del running[si]
          tag = "ok" if rc == 0 else f"FAILED(rc={rc})"
          print(f"[{time.strftime('%H:%M:%S')}] done    {point_name(grid[pidx], pidx)}  {tag}   ({len(results)}/{len(grid)})")
      time.sleep(2)
  finally:
    if mps_on:
      stop_mps()

  failed = [point_name(grid[i], i) for i, rc in results.items() if rc != 0]
  print(f"\nsweep complete: {len(results)-len(failed)}/{len(grid)} ok, {len(failed)} failed. logs in {logdir}/")
  if failed:
    print("  failed: " + ", ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))
    sys.exit(1)


if __name__ == "__main__":
  main()
