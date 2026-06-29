# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official implementation of **HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control** (RSS 2026). It trains a Unitree G1 humanoid to skateboard via RL in GPU-accelerated MuJoCo (mjlab / mujoco_warp), then deploys the policy as ONNX.

## Commands

All commands run through `uv` and **must be invoked from the repo root** — the env and AMP loader read `dataset/` via relative paths (`dataset/ref_pose/*.npy`, `dataset/skate_push/`).

```bash
# Install (mjlab is a pinned git dep; rsl-rl-lib is the local rsl_rl/ fork)
uv sync
uv pip install -e .

# Train. Registered task ids: Mjlab-Skater-Flat-Unitree-G1, Mjlab-Skater-Flat-Agibot-X2
uv run train Mjlab-Skater-Flat-Unitree-G1 --env.scene.num-envs 4096
uv run train Mjlab-Skater-Flat-Agibot-X2 --env.scene.num-envs 4096   # add --agent.amp-enabled False to skip AMP

# Play a checkpoint (.pt). --agent zero|random run dummy policies with no checkpoint.
uv run play Mjlab-Skater-Flat-Unitree-G1 --checkpoint_file <path-to-.pt>

# Standalone eval: run an ONNX policy in vanilla MuJoCo with keyboard steering
bash test_scene/sim.sh <path-to-.onnx> [g1|x2]   # x2 needs scripts/gen_scene_xml.py --robot x2 first

# Visualize the AMP push clips and the transition reference poses (render to video/)
bash test_scene/replay.sh --robot x2            # dataset/skate_push_x2 -> video/
bash test_scene/view_ref_pose.sh --robot x2     # dataset/ref_pose/x2_*.npy -> video/ref_pose/

# Lint (ruff is the configured linter; there is no unit-test suite)
uv run ruff check src
```

### Per-robot data generation (X2 is a scaffold — see the agibot-x2 memory)
The X2 reference poses and AMP push clips are generated, not authored:
`scripts/gen_ref_pose.py` (FK of the on-board stance → `dataset/ref_pose/x2_*.npy`),
`scripts/retarget_push_g1_to_x2.py` (joint-name remap of the G1 clips →
`dataset/skate_push_x2/`), and `scripts/gen_scene_xml.py` (the standalone
`test_scene/mjlab_scene_x2.xml` for sim.py). Both the poses and the retarget are
coarse and need tuning.

CLI parsing (both `train` and `play`) is two-stage tyro: the **first positional arg picks the task** from the registry, then remaining flags configure the dataclass with nested overrides — e.g. `--env.scene.num-envs 4096`, `--agent.max-iterations 60000`, `--video`. Multi-GPU is automatic: `--gpu-ids` defaults to `[0]`; pass a list or `all` and >1 GPU triggers a `torchrunx` launch (`MUJOCO_GL=egl` is set for you). Gradients are mean-reduced across ranks, so plain `--env.scene.num-envs N` is **per-GPU** — using >1 GPU that way multiplies the global batch (more data, diminishing returns), not the speed. To instead spend GPUs on **wall-clock** at a fixed batch, pass `--global-num-envs N`: it sets per-rank envs to `N // num_gpus`, so the effective batch matches a single-GPU run with `N` envs while collection is split across GPUs (`train.py::_apply_global_num_envs`). Speedup is bounded by env.step's fixed per-step CPU/sync overhead (~1.3× at 2 GPUs, asymptote ~1.8×).

Logs/checkpoints land in `logs/rsl_rl/<experiment_name>/<timestamp>/` (`experiment_name="g1_skater"`). `logs/`, `wandb/`, and `video/` are git-ignored.

## Architecture

### Two packages
- **`src/mjlab_husky/`** — the env + task framework, built on top of **mjlab** (manager-based RL env abstraction; a pinned git dependency). Uses 2-space indentation (mjlab convention).
- **`rsl_rl/`** — a **vendored fork** of `rsl_rl` installed as `rsl-rl-lib`, extended with AMP (Adversarial Motion Priors). Uses 4-space indentation (rsl_rl convention). `rsl_rl/build/` and `rsl_rl_lib.egg-info/` are build artifacts — ignore them; edit the top-level `rsl_rl/<module>` files.

### Task registration (automatic)
`tasks/__init__.py` calls mjlab's `import_packages(__name__, blacklist=["utils", ".mdp"])`, which recursively imports every package under `mjlab_husky.tasks`. The import side-effect in `tasks/skater/config/g1/__init__.py` calls `register_mjlab_task(...)`, populating the registry in `tasks/registry.py` (`_REGISTRY: task_id -> (env_cfg, play_env_cfg, rl_cfg, runner_cls)`). To add a robot/task: create a config package that imports cleanly and calls `register_mjlab_task`; `train`/`play` discover it via `list_tasks()`.

### Config layering (where to change behavior)
1. `tasks/skater/skater_env_cfg.py::make_g1_skater_env_cfg()` — robot-agnostic env: observation terms (policy vs. critic groups), actions, the `skate` command, domain-randomization events, the four reward dicts, and terminations.
2. `tasks/skater/config/g1/env_cfgs.py` — G1-specific overrides: contact sensors, action scale, the Bezier/SLERP body lists (`beizer_names`/`slerp_names`), `phase_ratios`, `steer_init_pos`, and **`play=True` overrides** (long episodes, no pushes, corruption off, eval_mode). `config/agibot_x2/` is the parallel package for the AgiBot X2 (same structure; also sets per-robot `push_ref_pose_path`/`steer_ref_pose_path`).
3. `tasks/skater/config/g1/rl_cfg.py` — PPO + AMP hyperparameters (`RslRlAMPOnPolicyRunnerCfg`).

### The phase-gated reward system (the core design)
`envs/g1_skate_rl_env.py::G1SkaterManagerBasedRlEnv` overrides mjlab's base env. A skating stroke is one cycle of `cycle_time` (6s) split by `phase_ratios = [0.0, 0.4, 0.5, 0.95, 1.0]` into four phases: **push → push2steer → steer → steer2push**. `_resample_contact_phases()` computes the current phase from `phase_length_buf` and writes a one-hot-ish `contact_phase` (columns = push, steer, push2steer, steer2push).

The env holds **four separate `RewardManager`s** — `push_reward_manager`, `steer_reward_manager`, `transition_reward_manager`, `reg_reward_manager` (built in `load_managers()` from the four `*_rewards` dicts on the cfg). In `step()`, each manager's reward is **masked by its `contact_phase` column** so only the active phase contributes; the regularization reward is always on. Reward functions live in `tasks/skater/mdp/rewards.py`, grouped by phase with `push_*` / `steer_*` / `transition_*` / `reg_*` prefixes.

During the two transition phases, target whole-body poses are interpolated between a runtime-captured start pose and a reference end pose (`dataset/ref_pose/{push,steer}_start_pose_b.npy`) using a quadratic **Bezier curve for positions** and **quaternion SLERP for rotations** (`bezier_curve`, `quaternion_slerp` at the bottom of the env file); `transition_*` rewards track these. Poses are expressed in the skateboard's body frame.

### AMP (push phase only)
`rsl_rl/runners/amp_on_policy_runner.py` drives training (`AMPOnPolicyRunner`, algorithm `AMP_PPO`). A `DiscriminatorMulti` learns to distinguish policy motion from human push mocap (`dataset/skate_push/human_push_*.npy`, loaded by `G1_AMPLoader`). The AMP-style reward **only replaces the task reward where `contact_phase[:,0]==1` (push phase)** — see the `mask` logic in `learn()`. AMP observations are the robot joint positions (`env.get_amp_observations()` → `robot.data.joint_pos`, 23-dim for G1 / 31 for X2, 5-frame history). The loader is robot-agnostic via `amp_obs_slices` (which clip columns form the AMP obs); `amp_enabled=False` skips the discriminator + loader entirely so the push phase trains on task rewards only (no clips needed). The X2 has no published push mocap, so its clips are a coarse joint-name remap of the G1 clips (`scripts/retarget_push_g1_to_x2.py`, `rsl_rl/utils/retarget.py`).

### Deployment / standalone eval
`SkaterOnPolicyRunner` (`tasks/skater/rl/runner.py`) exports an ONNX policy on every save when logging to wandb. `test_scene/sim.py` runs that ONNX in **plain MuJoCo** (no mjlab/warp), reading a scene XML (`test_scene/mjlab_scene.xml` for G1; generate the X2 one with `scripts/gen_scene_xml.py --robot x2`), with arrow-key steering. `_derive_layout()` derives the robot-specific pieces — `num_actions`, the joint↔ctrl `reindex_list`, per-joint `action_scale` (`0.25·effort/stiffness` from the actuator params), and the init pose (scene keyframe) — **from the scene model**, so the same code drives G1 and X2 (validated to reproduce the old hand-tuned G1 constants). The remaining hand-rolled part is the **obs term order/scaling** (`obs_block_dims`, `history_len=5`, the `obs_proprio` concat), which must still mirror the policy obs group in `skater_env_cfg.py` — change that layout and you must update `sim.py` to match.

### Real-hardware deployment (`deploy/`)
`deploy/agibot_x2_ultra_deploy/` is a **git submodule** of the upstream AgiBot Deploy Host (a metadata-driven, policy-agnostic ONNX/ROS 2 runtime for X2 hardware) — vendored read-only. `deploy/husky_x2_deploy/` is **our adapter**: the host's generic `ObsBuilder` assembles a single obs frame, but the skater policy needs the 5-frame term-major history, a `heading` term, a scalar `phase` clock, and the `[v,h]` command, so the adapter adds `SkateObsAssembler` + `SkateCommandProvider` and a `SkateDeployNode` that subclasses the host node (reusing its soft-start/E-stop/teleop/publish machinery). The host already does the action mapping (`q = a·scale + default`, joint-name → SDK group split) identically to `sim.py`. The skater ONNX already embeds joint_names/gains/`default_joint_pos`/`action_scale`/`observation_names`; `husky-skate-export` only adds the sidecar (`actor_obs_spec`, skater `command_spec`, `input_contract`). `husky-skate-validate` is the offline gate — it asserts the deploy obs/action reproduce `sim.py` **bit-exact** (run it before any robot run). The obs term order/scales there must stay in sync with `skater_env_cfg.py` and `sim.py` (`deploy/husky_x2_deploy/husky_x2_deploy/export_skater_deploy.py::TERM_SCALE`). See `deploy/README.md`.

## Gotchas
- The `play` env cfg is a separate registered config (`play_env_cfg`), not the train cfg — overrides for evaluation belong in the `if play:` block of `env_cfgs.py`.
- Tracked checkpoints in `ckpts/` (`test.pt`, `test.onnx`) are reference artifacts referenced by the README, not throwaway.
- Several config fields are misspelled consistently (`beizer_names`, `_set_skatedboard_joint_pos`); match the existing spelling rather than "fixing" it, since names are looked up by string.
