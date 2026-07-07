"""Watchdog for a fine-tune run: polls tensorboard scalars against calibrated tripwires.

Run alongside scripts/finetune.py (see FINETUNING.md). Checks every 5 minutes and
exits with a distinct code so a supervisor (human or agent) gets a useful signal:

  exit 0 -> training reached its final iteration (completed)
  exit 2 -> an ABORT threshold was sustained (details printed; training is LEFT
            RUNNING — this tool never kills anything)
  exit 3 -> stall: no new scalars for --stall-min minutes

Thresholds are means over the last 50 logged iters, guarded by a warmup window
(the value-loss spike and episode-length deque refill in the first ~100 iters are
expected and benign). Defaults were calibrated on the 2026-07 deck-pinch fine-tune;
override per-run as needed. The entropy-ratchet alert fires when the action std
climbs faster than --ratchet-rate per iter — the signature failure mode of
fine-tuning a converged policy (see FINETUNING.md).

Usage (from repo root, after launching a fine-tune):
    uv run python scripts/finetune_watchdog.py --total-iters 1500
    # or pin an explicit run dir:
    uv run python scripts/finetune_watchdog.py --run-dir logs/rsl_rl/x2_skater/<ts> --total-iters 1500
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(status_path: str, msg: str) -> None:
  line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
  print(line, flush=True)
  with open(status_path, "a") as f:
    f.write(line + "\n")


def list_run_dirs(log_root: str) -> set[str]:
  return {d for d in glob.glob(os.path.join(log_root, "*", "20*")) if os.path.isdir(d)}


def load_scalars(run_dir: str):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags().get("scalars", [])

  def series(sub: str):
    m = [t for t in tags if sub in t]
    if not m:
      return None
    return [(e.step, e.value) for e in ea.Scalars(m[0])]

  return {
    "std": series("mean_noise_std"),
    "eplen": series("mean_episode_length"),
    "disc": series("mean_step_disc_reward"),
    "vloss": series("value_function"),
  }


def tail_mean(s, n=50):
  if not s:
    return None
  vals = [v for _, v in s[-n:]]
  return sum(vals) / len(vals)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--run-dir", default=None,
                  help="run dir to watch; default: newest under logs/rsl_rl/*/ (started recently)")
  ap.add_argument("--total-iters", type=int, required=True,
                  help="the --iters value passed to finetune.py (completion detection)")
  ap.add_argument("--poll-s", type=int, default=300)
  ap.add_argument("--stall-min", type=int, default=90)
  ap.add_argument("--warmup-iters", type=int, default=100)
  ap.add_argument("--std-abort", type=float, default=0.20,
                  help="abort if mean action std exceeds this")
  ap.add_argument("--ratchet-rate", type=float, default=2e-4,
                  help="alert if std climbs faster than this per iter (entropy ratchet)")
  ap.add_argument("--disc-min", type=float, default=0.15,
                  help="abort if per-step AMP disc reward falls below this (skipped if absent)")
  ap.add_argument("--eplen-min", type=float, default=120.0)
  ap.add_argument("--vloss-max", type=float, default=5.0,
                  help="post-warmup sustained value-loss abort (100 aborts anytime)")
  args = ap.parse_args()

  status_path = os.path.join(REPO_ROOT, "logs", "finetune_watchdog.log")
  os.makedirs(os.path.dirname(status_path), exist_ok=True)
  launch_t = time.time()

  # auto-discovery: accept only a run dir that STARTED recently — either one that
  # appears after we began watching, or a preexisting one whose params/agent.yaml
  # (written exactly once at run start) is fresh. Plain "newest mtime" would latch
  # onto any concurrent run, since every checkpoint save bumps a dir's mtime.
  log_root = os.path.join(REPO_ROOT, "logs", "rsl_rl")

  def started_at(d: str) -> float | None:
    marker = os.path.join(d, "params", "agent.yaml")
    return os.path.getmtime(marker) if os.path.exists(marker) else None

  run_dir = args.run_dir
  if run_dir is None:
    preexisting = list_run_dirs(log_root)
    while run_dir is None:
      fresh = [d for d in list_run_dirs(log_root)
               if d not in preexisting
               or ((t := started_at(d)) is not None and t > launch_t - 900)]
      if fresh:
        run_dir = max(fresh, key=lambda d: started_at(d) or 0.0)
      else:
        log(status_path, "waiting for a freshly-started run dir (pass --run-dir to pin one)...")
        time.sleep(60)
        if time.time() - launch_t > 1800:
          log(status_path, "STALL: no new run dir appeared within 30 min")
          sys.exit(3)
  log(status_path, f"watching: {run_dir}")

  start_iter = None
  last_iter, last_progress_t = -1, time.time()
  while True:
    try:
      s = load_scalars(run_dir)
    except Exception as e:  # noqa: BLE001 — keep watching through transient read errors
      log(status_path, f"scalar read error ({e}); retrying")
      time.sleep(args.poll_s)
      continue

    cur = s["std"][-1][0] if s["std"] else -1
    if start_iter is None and s["std"]:
      start_iter = s["std"][0][0]
    if cur > last_iter:
      last_iter, last_progress_t = cur, time.time()

    # completion: the runner names the final ckpt by the last COMPLETED index
    # (start + total - 1), so detect via scalars or checkpoints reaching it.
    if start_iter is not None:
      end_iter = start_iter + args.total_iters - 1
      ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
      maxck = max((int(p.split("_")[-1].split(".")[0]) for p in ckpts), default=-1)
      if cur >= end_iter or maxck >= end_iter:
        log(status_path, f"DONE: iter {cur}, final ckpt model_{maxck}.pt")
        sys.exit(0)

    if time.time() - last_progress_t > args.stall_min * 60:
      log(status_path, f"STALL: no progress for {args.stall_min} min (iter {last_iter})")
      sys.exit(3)

    std, eplen = tail_mean(s["std"]), tail_mean(s["eplen"])
    disc, vloss = tail_mean(s["disc"]), tail_mean(s["vloss"], 10)
    past_warmup = start_iter is not None and cur > start_iter + args.warmup_iters
    aborts, alerts = [], []
    if std is not None:
      if std > args.std_abort:
        aborts.append(f"std {std:.3f}>{args.std_abort} ")
      elif s["std"] and len(s["std"]) > 50:
        (i0, v0), (i1, v1) = s["std"][-50], s["std"][-1]
        rate = (v1 - v0) / max(1, i1 - i0)
        if rate > args.ratchet_rate:
          alerts.append(f"entropy ratchet: std +{rate:.1e}/iter (cut entropy_coef)")
    # skip the disc check for AMP-disabled runs: the tag is written unconditionally
    # (zeros when amp_enabled=False), so "series exists" is not "AMP is on".
    amp_active = s["disc"] is not None and any(v > 0 for _, v in s["disc"])
    if past_warmup and amp_active and disc is not None and disc < args.disc_min:
      aborts.append(f"disc reward {disc:.3f}<{args.disc_min}")
    if past_warmup and eplen is not None and eplen < args.eplen_min:
      aborts.append(f"ep len {eplen:.0f}<{args.eplen_min:.0f}")
    if vloss is not None and (vloss > 100 or (past_warmup and vloss > args.vloss_max)):
      aborts.append(f"value loss {vloss:.2f}")

    done = (cur - start_iter) if start_iter is not None else -1
    log(status_path,
        f"iter {cur} ({done}/{args.total_iters}) std={std and f'{std:.4f}'} "
        f"eplen={eplen and f'{eplen:.0f}'} disc={disc and f'{disc:.3f}'} "
        f"vloss={vloss and f'{vloss:.3f}'}"
        + (f"  ALERTS: {alerts}" if alerts else "")
        + (f"  *** ABORT: {aborts}" if aborts else ""))
    if aborts:
      log(status_path, "ABORT thresholds sustained (training left running) — "
                       "bisect checkpoints with scripts/eval_gate.py and see FINETUNING.md")
      sys.exit(2)
    time.sleep(args.poll_s)


if __name__ == "__main__":
  main()
