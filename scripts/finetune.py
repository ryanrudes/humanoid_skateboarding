"""Fine-tune a trained skater policy with safe, battle-tested defaults.

Wraps ``uv run train`` with the settings that made the 2026-07 deck-pinch fix work
(see FINETUNING.md for the full rationale + monitoring guide):

  * RESUME (not warm-start): keeps the optimizer, AMP discriminator, amp_normalizer,
    and obs-normalizers — the slow state that protects the skills you already have.
  * LR pinned to the donor's annealed value (read from the checkpoint's optimizer
    state). Since 89651c8 the runner also restores this itself on resume, so the
    pin is redundant insurance — kept because it is free and self-documenting.
  * LOW entropy (default 0.001): a converged policy barely resists the entropy
    bonus, so the default 0.005 ratchets the action std up without bound
    (~+2e-4/iter at 4096 envs) until noise degrades every phase.
  * Dense checkpointing (default every 50 iters) so any drift is a bisect-and-
    rollback, never a loss. The donor checkpoint is only ever COPIED.

Usage (from repo root):
    uv run python scripts/finetune.py Mjlab-Skater-Flat-Agibot-X2 model_307000.pt \
        --iters 1500 [--num-envs 4096] [--gpu 0] [--dry-run]

Anything after ``--`` is passed through to ``train`` verbatim, e.g.:
    ... -- --agent.algorithm.desired-kl 0.02

Pair with:  scripts/finetune_watchdog.py  (threshold monitor, run alongside)
            scripts/eval_gate.py          (rollout gate for any checkpoint)
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("task", help="registered task id, e.g. Mjlab-Skater-Flat-Agibot-X2")
  ap.add_argument("checkpoint", help="path to the donor .pt checkpoint (any location)")
  ap.add_argument("--iters", type=int, default=1500,
                  help="fine-tune iterations (ADDITIVE: counter continues from the donor's)")
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--entropy-coef", type=float, default=0.001,
                  help="LOW on purpose — see the entropy-ratchet section of FINETUNING.md")
  ap.add_argument("--lr", type=float, default=None,
                  help="initial LR; default = the donor's annealed LR read from the checkpoint")
  ap.add_argument("--save-interval", type=int, default=50)
  ap.add_argument("--gpu", default=None, help="sets CUDA_VISIBLE_DEVICES")
  ap.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"],
                  help="wandb (default) logs to the cloud AND writes local TB events, so "
                       "the watchdog keeps working; use tensorboard for throwaway local "
                       "trials (retro-sync later with scripts/sync_tb_to_wandb.py)")
  ap.add_argument("--dry-run", action="store_true",
                  help="stage + print the command and monitoring cheat-sheet, don't launch")
  args, extra = ap.parse_known_args()
  if extra and extra[0] == "--":
    extra = extra[1:]

  ckpt = Path(args.checkpoint).resolve()
  if not ckpt.exists():
    sys.exit(f"checkpoint not found: {ckpt}")

  # -- read the donor's annealed LR + iteration counter -------------------------
  donor_iter, donor_lr = None, None
  if args.lr is None or True:  # always read iter for the cheat-sheet
    import torch  # deferred: slow import

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    donor_iter = d.get("iter")
    try:
      donor_lr = d["optimizer_state_dict"]["param_groups"][0]["lr"]
    except (KeyError, IndexError, TypeError):
      donor_lr = None
    del d
  lr = args.lr if args.lr is not None else (donor_lr if donor_lr is not None else 1e-5)

  # -- resolve the experiment dir from the task's RL cfg ------------------------
  import mjlab.tasks  # noqa: F401  (registration side-effects)
  import mjlab_husky.tasks  # noqa: F401
  from mjlab_husky.tasks.registry import load_rl_cfg

  exp_name = load_rl_cfg(args.task).experiment_name

  # -- stage the donor into a fresh, uniquely-named run dir ---------------------
  # (resume resolves <logs/rsl_rl/<experiment>>/<load_run>/<checkpoint>, and
  #  load_run is regex-matched against dir names — a unique stage dir avoids
  #  ever matching someone else's run.)
  stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  stage = REPO_ROOT / "logs" / "rsl_rl" / exp_name / f"ft_stage_{stamp}"
  stage.mkdir(parents=True, exist_ok=False)
  shutil.copy2(ckpt, stage / ckpt.name)

  cmd = [
    "uv", "run", "train", args.task,
    "--env.scene.num-envs", str(args.num_envs),
    "--agent.resume", "True",
    "--agent.load-run", stage.name,
    "--agent.load-checkpoint", ckpt.name,
    "--agent.max-iterations", str(args.iters),
    "--agent.save-interval", str(args.save_interval),
    "--agent.algorithm.learning-rate", f"{lr:g}",
    "--agent.algorithm.entropy-coef", f"{args.entropy_coef:g}",
    "--agent.logger", args.logger,
    *extra,
  ]
  env = dict(os.environ)
  env.setdefault("MUJOCO_GL", "egl")
  if args.gpu is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  if args.logger == "tensorboard":
    env.setdefault("WANDB_MODE", "disabled")

  final_iter = (donor_iter + args.iters - 1) if donor_iter is not None else None
  print("=" * 78)
  print(f"fine-tune: {ckpt.name}  (donor iter {donor_iter})  ->  +{args.iters} iters"
        + (f"  (final ckpt ~ model_{final_iter}.pt)" if final_iter else ""))
  print(f"staged at: {stage}")
  print(f"LR {lr:g} ({'donor annealed' if args.lr is None and donor_lr else 'override'})"
        f"  entropy_coef {args.entropy_coef:g}  save every {args.save_interval}")
  print("-" * 78)
  print("monitor (tensorboard tags — healthy bands from the 2026-07 calibration):")
  print("  Policy/mean_noise_std          expect ~flat/slow drift; ratchet = climbing >2e-4/iter")
  print("  Train/mean_episode_length      should rise or hold; <120 sustained = push failing")
  print("  Train/mean_step_disc_reward    (AMP tasks) hold >=0.27; <0.15 sustained = corruption")
  print("  Loss/value_function            spike to ~16-53 in first ~10 iters is NORMAL; then <1")
  print("  Episode_Reward/* are EPISODE SUMS — divide by episode length before comparing!")
  print(f"  watchdog:  uv run python scripts/finetune_watchdog.py --total-iters {args.iters}")
  print(f"  eval gate: uv run python scripts/eval_gate.py {args.task} <run_dir>/model_*.pt")
  print("=" * 78)
  print(" ".join(cmd))
  if args.dry_run:
    print("(dry run: not launching; stage dir left in place)")
    return
  sys.exit(subprocess.run(cmd, env=env, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
  main()
