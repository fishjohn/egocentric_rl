# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

from legged_gym import LEGGED_GYM_ENVS_DIR, WANDB_SAVE_DIR


def setup_wandb(args):
    log_pth = WANDB_SAVE_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth, exist_ok=True)
    except:
        raise RuntimeError(f"Failed to create directory '{log_pth}'")
    if args.debug:
        mode = "disabled"
        args.num_envs = 64
    else:
        mode = "online"

    if args.no_wandb:
        mode = "disabled"
    wandb.init(project=args.proj_name, name=args.exptid, group=args.exptid[:3], mode=mode,
               dir=os.path.join(WANDB_SAVE_DIR, "logs"))
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/biped/biped.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/biped/biped_config.py", policy="now")

    return args, log_pth


def train(args):
    args, log_pth = setup_wandb(args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=log_pth)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
