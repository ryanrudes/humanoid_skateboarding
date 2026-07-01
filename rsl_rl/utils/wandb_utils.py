# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb. If WANDB_RUN_ID is set (+ optional WANDB_RESUME), append to
        # that existing run so a resumed training run continues the SAME wandb run (one
        # continuous curve) instead of forking a new run. Unset => fresh run as before.
        resume_id = os.environ.get("WANDB_RUN_ID")
        wandb.init(
            project=project,
            entity=entity,
            name=None if resume_id else run_name,  # keep the original run's name on resume
            id=resume_id,
            resume="allow" if resume_id else None,
        )

        # Add log directory to wandb (allow_val_change: the log_dir differs on resume).
        wandb.config.update({"log_dir": log_dir}, allow_val_change=True)

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        # allow_val_change so re-logging config on a resumed run (bumped max_iterations,
        # new log_dir, etc.) updates the values instead of raising.
        wandb.config.update({"runner_cfg": runner_cfg}, allow_val_change=True)
        wandb.config.update({"policy_cfg": policy_cfg}, allow_val_change=True)
        wandb.config.update({"alg_cfg": alg_cfg}, allow_val_change=True)
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()}, allow_val_change=True)
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)}, allow_val_change=True)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
