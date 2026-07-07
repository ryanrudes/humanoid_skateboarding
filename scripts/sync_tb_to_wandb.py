"""Retro-sync a tensorboard-only training run to Weights & Biases.

Replays every scalar series from a run dir's tfevents file into a fresh wandb run
(steps preserved, so resumed-iteration runs chart on the global iteration axis) and
attaches the run's params/agent.yaml + env.yaml as the run config.

Usage (from repo root, once per run):
    uv run python scripts/sync_tb_to_wandb.py logs/rsl_rl/x2_skater/<run_dir> \
        --name ft-pinchfix-phase1 --notes "resume of model_307000 on capsule feet" \
        [--project Skateboarding] [--tags finetune,retro-sync] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("run_dir", help="run dir containing tfevents + params/")
  ap.add_argument("--name", required=True, help="wandb run name")
  ap.add_argument("--project", default="Skateboarding")
  ap.add_argument("--notes", default="")
  ap.add_argument("--tags", default="finetune,retro-sync")
  ap.add_argument("--dry-run", action="store_true", help="list tags/steps, upload nothing")
  args = ap.parse_args()

  run_dir = Path(args.run_dir).resolve()
  if not run_dir.exists():
    raise SystemExit(f"run dir not found: {run_dir}")

  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags().get("scalars", [])
  if not tags:
    raise SystemExit(f"no scalar tags found under {run_dir}")

  # merge all series into one dict per step so each step is a single wandb.log call
  by_step: dict[int, dict[str, float]] = defaultdict(dict)
  for tag in tags:
    for ev in ea.Scalars(tag):
      by_step[ev.step][tag] = ev.value
  steps = sorted(by_step)
  print(f"{run_dir.name}: {len(tags)} tags, {len(steps)} steps "
        f"[{steps[0]}..{steps[-1]}]")
  if args.dry_run:
    for t in sorted(tags):
      print("  ", t)
    return

  config = {}
  try:
    import yaml

    for f in ("agent.yaml", "env.yaml"):
      p = run_dir / "params" / f
      if p.exists():
        config[f.split(".")[0]] = yaml.unsafe_load(p.read_text())
  except Exception as e:  # noqa: BLE001 — config attachment is best-effort
    print(f"(config attach skipped: {e})")

  os.environ.pop("WANDB_MODE", None)  # the run may have been launched with it disabled
  import wandb

  run = wandb.init(
    project=args.project,
    name=args.name,
    notes=args.notes,
    tags=[t for t in args.tags.split(",") if t],
    config=config,
    settings=wandb.Settings(init_timeout=120),
  )
  for s in steps:
    wandb.log(by_step[s], step=s)
  url = run.url
  run.finish()
  print(f"synced -> {url}")


if __name__ == "__main__":
  main()
