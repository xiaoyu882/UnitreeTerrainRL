import os
import numpy as np
from datetime import datetime
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# def train(args):
#     env, env_cfg = task_registry.make_env(name=args.task, args=args)
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
#     ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    train_type = getattr(env_cfg.env, "train_type", "")

    if train_type == "MorAL":
        from rl.MorAL.runners.policy_runner import MorALPolicyRunner
        from rl.MorAL.configs.go2_moral_train_cfg import train_cfg  # 你需要建这个 dict 文件

        log_dir = os.path.join("logs", args.task, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

        runner = MorALPolicyRunner(env, train_cfg, log_dir=log_dir, device=env.device)
        runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"],
                     init_at_random_ep_len=True)
        return

    # 否则默认走原 rsl_rl
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
