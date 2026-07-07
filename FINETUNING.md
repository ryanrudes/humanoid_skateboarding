# Fine-tuning a trained skater policy

How to adapt an already-good checkpoint after an env/physics/asset change (new collision
geometry, tweaked board, retuned actuator, …) **without losing the skills it already has**.
Everything here was derived and empirically validated during the 2026-07 "deck-pinch" fix,
where `model_307000.pt` (X2) was fine-tuned past a collision-geometry change in **~2k
iterations (~2 h)** instead of a ~50k-iteration retrain, with zero measurable forgetting.

## TL;DR

```bash
# 1. launch with safe defaults (stages the ckpt, pins LR, low entropy, dense ckpts)
uv run python scripts/finetune.py Mjlab-Skater-Flat-Agibot-X2 model_307000.pt --iters 1500

# 2. in a second shell: threshold watchdog (exits loudly on abort/stall/done)
uv run python scripts/finetune_watchdog.py --total-iters 1500

# 3. gate any checkpoint (also used to bisect after an abort)
uv run python scripts/eval_gate.py Mjlab-Skater-Flat-Agibot-X2 \
    logs/rsl_rl/x2_skater/<run>/model_<N>.pt --gate
```

## When to fine-tune vs retrain

Fine-tune when the failure is **localized** (one phase / one transition breaks after the
change) and the reward already points at the behavior you want. Retrain when the reward
itself was wrong (the old optimum was a *reward* exploit, not a physics exploit), or the
obs/action space changed (then see `--warm-start-checkpoint`).

Litmus test used for the pinch fix: read the reward terms for the broken phase and ask
"would the correct behavior score >= the broken one under the new physics?" If yes, the
survival gradient flips to your side and fine-tuning converges fast with **no reward
changes**.

## The four knobs that matter (and why)

| Knob | Setting | Why |
|---|---|---|
| Loading | **`--agent.resume`, not warm-start** | Resume restores the optimizer moments, AMP discriminator + `amp_normalizer`, and actor/critic obs-normalizers (whose ~2e10 sample counts make them effectively frozen). Env-side state is NOT checkpointed (extra keys in older checkpoints, e.g. `env_state`, are ignored by `load()`). Warm-start (`--warm-start-checkpoint`) resets the discriminator — it re-warms against your *good* phase and degrades exactly the skill you want to keep. |
| Learning rate | **pin to the donor's annealed LR** (`finetune.py` reads it from the checkpoint; typically the 1e-5 floor) | Historically, resume reset the LR from the cfg (1e-3) and `update()` clobbered every param group on the first minibatch — ~20 Adam steps at 100× the trained LR against a maximally-stale critic (observed wrecking a resumed policy: reward 179→2). **Fixed at the source in 89651c8**: `load()` now restores `self.alg.learning_rate` from the loaded optimizer's param groups, so resume is seamless. `finetune.py` still passes the donor LR explicitly as redundant insurance (harmless — both paths agree). |
| Entropy | **LOW: 0.001** (not the training default 0.005, and never "bumped for exploration") | **The entropy-std ratchet is the signature failure mode of fine-tuning a converged policy.** The action std is a state-independent learned parameter (one value per joint; `Policy/mean_noise_std` is its mean) shared across all phases; a converged policy's objective barely resists the entropy bonus, so std climbs linearly without plateau (measured: +2e-4/iter at 4096 envs — 0.08 → 0.17 in 550 iters; extrapolates to a destructive ~0.6 by 3k). Exploration is *not* needed anyway: the dense tracking rewards supply the gradient, and the collapsed std self-inflates even at low coefficients. If std still ratchets at 0.001, cut further or set 0. |
| Checkpointing | **`--agent.save-interval 50`** + never touch the donor file | Converts any slow drift into a bisect-and-rollback (`eval_gate.py --gate` over the 50-iter checkpoints). Forgetting becomes recoverable, not catastrophic. |

## Why this codebase is inherently forgetting-resistant (verified)

- **Train-mode terminations are the anti-forgetting mechanism.** `fell_over` (70° tilt),
  `feet_off_board` (<5 N both feet, 2 steps), `illegal_contact` end a failed episode within
  ~2–30 steps, so the rollout buffer stays ~85–90% healthy on-distribution rehearsal of the
  intact skill. Don't weaken them for a fine-tune. (Play cfg strips these — never judge
  train data mix from play behavior.)
- **Actor and critic are separate MLPs** — a stale critic biases advantages but can't
  corrupt actor features directly. Expect `Loss/value_function` to spike (16–53) in the
  first ~10 iters and decay below ~1 by iter ~50. That transient is normal.
- **AMP is an additive, clamped bonus** — `reward = task + 0.02·amp_reward_coef·clamp(...)`
  ≤ 0.1/step (`use_lerp=False` branch, `rsl_rl/modules/discriminator_multi.py`), masked to
  the push phase. Its blast radius is small, and a healthy discriminator *helps* preserve
  the motion style during the fine-tune.
- **Phase clock is not randomized on reset** (`phase_length_buf` zeroed): phases *after* the
  broken one get ~0% data until the fix lands. Their skills are protected by absence of
  gradient — but also invisible. Eval them at the first gate after `time_out` terminations
  appear.

## Canary protocol (do this before committing to a long run)

One 200-iter run at 2048 envs (~20–30 min) answers the forgetting question empirically:
`scripts/finetune.py <task> <ckpt> --iters 200 --num-envs 2048`, then `eval_gate.py` the
final checkpoint and compare training curves. What "healthy" looked like in the calibration
run: per-step push reward flat, `mean_step_disc_reward` steady-or-rising, episode length
rising, broken-phase tracking rewards rising several-fold. If the canary is clean, launch
the real run; if not, tighten the trust region (`--agent.algorithm.desired-kl 0.005
--agent.algorithm.clip-param 0.1` via passthrough args after `--`).

Runs log to **wandb by default** (and always write local TB events, which is what
the watchdog and the bands below read). A tensorboard-only trial can be uploaded
after the fact with `scripts/sync_tb_to_wandb.py <run_dir> --name <name>`.

## Monitoring cheat-sheet (tensorboard tags; calibrated healthy bands)

| Tag | Healthy | Act when |
|---|---|---|
| `Policy/mean_noise_std` | ~flat, slow drift | climbing >2e-4/iter = **ratchet → cut entropy_coef, resume from last good ckpt**; >0.2 abort |
| `Train/mean_episode_length` | rising / high | <120 sustained = the *intact* phase is failing (most direct forgetting alarm) |
| `Train/mean_step_disc_reward` | ≥0.27 | <0.15 sustained = AMP corruption of the push signal |
| `Loss/value_function` | spike→<1 by ~iter 50 | >100 anytime, or not decaying — stop and inspect |
| `Episode_Reward/*` | — | **These are per-episode sums scaled by the constant max episode length** (sum / 20 s): longer episodes still inflate them mechanically and dilute per-step phase terms. Divide by realized episode length before comparing. |

Also expect a **second watch-window when the fix starts working**: the data distribution
shifts qualitatively (new phases get samples, episode structure changes). Keep the eval
gate running through that transition.

## Worked example (the deck-pinch fix, 2026-07)

Foot collision changed from 12 sparse spheres to a continuous capsule sole → the trained
policy's back-foot "deck-pinch" stance became impossible → it rode the push phase perfectly
and fell at every first mount. Fine-tune: resume `model_307000.pt`, rewards unchanged.
Phase 1 (entropy 0.005, 550 iters): mount fixed, but std ratcheted 0.08→0.17 and the
watchdog aborted. Phase 2 (resume from `model_307550`, entropy 0.001, 1500 iters): ratchet
reversed (0.175→0.158), episode length →~930/1000, `feet_off_board` terminations
5.5→0.42/iter. Final `model_309049.pt`: 1000-step eval, never falls, 0 below-deck steps,
repeat mounts working. Total: 2,049 iters. Push-phase metrics never dipped at any point.
