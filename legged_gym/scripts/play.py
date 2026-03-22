# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os
# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

# import numpy as np
# import torch


# def play(args):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override some parameters for testing
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     env_cfg.env.test = True

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     obs = env.get_observations()
#     # load policy
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device=env.device)
    
#     # export policy as a jit module (used to run it from C++)
#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#         print('Exported policy as jit script to: ', path)

#     for i in range(10*int(env.max_episode_length)):
#         actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())

# if __name__ == '__main__':
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     args = get_args()
#     play(args)

import os
import isaacgym  # noqa
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa
from legged_gym.utils import get_args, task_registry

from rl.MorAL.runners.policy_runner import MorALPolicyRunner

# --- Force MorALPolicyRunner to use MorAL ActorCritic (avoid name conflict with rsl_rl) ---
import rl.MorAL.runners.policy_runner as pr
from rl.MorAL.modules.actor_critic import ActorCritic as MorALActorCritic

pr.ActorCritic = MorALActorCritic  # make eval("ActorCritic") resolve to MorAL version
# ----------------------------------------------------------------------------------------


def _latest_run_dir(task_name: str) -> str:
    task_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", task_name)
    if not os.path.isdir(task_dir):
        raise RuntimeError(f"log dir not found: {task_dir}")

    runs = [os.path.join(task_dir, d) for d in os.listdir(task_dir)]
    runs = [d for d in runs if os.path.isdir(d)]
    if not runs:
        raise RuntimeError(f"no runs under: {task_dir}")

    runs.sort(key=lambda p: os.path.getmtime(p))
    return runs[-1]


def _latest_checkpoint(task_name: str) -> str:
    run_dir = _latest_run_dir(task_name)
    ckpt = os.path.join(run_dir, "stage1_nn", "last.pt")
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"checkpoint not found: {ckpt}")
    return ckpt


def _infer_moral_cfg_from_ckpt(ckpt_path: str, num_obs: int) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["actor_state_dict"]

    def linear_out_dims(prefix: str):
        idxs = []
        for k in sd.keys():
            if k.startswith(prefix) and k.endswith(".weight"):
                # e.g. actor.0.weight -> idx=0
                parts = k.split(".")
                if len(parts) >= 3 and parts[1].isdigit():
                    idxs.append(int(parts[1]))
        idxs = sorted(set(idxs))
        outs = [sd[f"{prefix}.{i}.weight"].shape[0] for i in idxs]
        ins0 = sd[f"{prefix}.{idxs[0]}.weight"].shape[1]
        ins_last = sd[f"{prefix}.{idxs[0]}.weight"].shape[1]
        return idxs, outs, ins0, ins_last

    # actor hidden dims: all Linear outs except last (which is num_actions)
    a_idxs, a_outs, a_in0, _ = linear_out_dims("actor")
    actor_hidden_dims = a_outs[:-1]
    num_actions = a_outs[-1]

    # critic hidden dims: all Linear outs except last (which is 1)
    c_idxs, c_outs, c_in0, _ = linear_out_dims("critic")
    critic_hidden_dims = c_outs[:-1]

    # priv_info_dim inferred from critic first layer input: num_obs + priv_info_dim
    priv_info_dim = int(c_in0 - num_obs)
    if priv_info_dim <= 0:
        raise RuntimeError(f"invalid priv_info_dim inferred: {priv_info_dim} (critic_in={c_in0}, num_obs={num_obs})")

    # encoder_mlp_units inferred from dm_encoder in ActorCritic
    # keys like: dm_encoder.encoder.0.weight, .2.weight, .4.weight
    enc0 = sd["dm_encoder.encoder.0.weight"].shape[0]
    enc2 = sd["dm_encoder.encoder.2.weight"].shape[0]
    enc4 = sd["dm_encoder.encoder.4.weight"].shape[0]
    encoder_mlp_units = [int(enc0), int(enc2), int(enc4)]

    cfg = {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "save_interval": 200,
            "export_policy": False,
        },
        "algorithm": {},
        "policy": {
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "activation": "elu",
            "init_noise_std": float(sd["std"].mean().item()) if "std" in sd else 1.0,
        },
        "Encoder": {
            "priv_info": True,
            "priv_info_dim": priv_info_dim,
            "HistoryLen": 4,
            "encoder_mlp_units": encoder_mlp_units,
            "proprio_adapt": False,
            "checkpoint_model": None,
            "proprio_adapt_out_dim": 11,
            "morph_priv_info": True,
            "Hist_info_dim": num_obs * 4,
        },
        # MorALPolicyRunner 里还会读取 train_cfg["Encoder"]["encoder_mlp_units"] 等
    }
    return cfg


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    print("clip_actions:", env.cfg.normalization.clip_actions)
    print("action_scale:", env.cfg.control.action_scale)
    print("control_type:", env.cfg.control.control_type)

    ckpt = _latest_checkpoint(args.task)
    log_dir = os.path.dirname(os.path.dirname(ckpt))  # .../<run>

    train_cfg_dict = _infer_moral_cfg_from_ckpt(ckpt, num_obs=env.num_obs)

    runner = MorALPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device=env.device)
    runner.load(ckpt, load_optimizer=False)
    sd = torch.load(ckpt, map_location="cpu")["actor_state_dict"]

    policy, _ = runner.get_inference_policy(device=env.device)

    obs_dict = env.get_observations()
    for _ in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs_dict)
      
            # ====== 用缩放代替硬 clamp ======
            g = 0.25  # 先从 0.25 开始试；不稳就 0.2 / 0.15；太保守就 0.3 / 0.4
            actions = actions * g

            clip = env.cfg.normalization.clip_actions  # 用原来 cfg 的 clip
            actions = torch.clamp(actions, -clip, clip)
            # =================================

        out = env.step(actions)

        if isinstance(out, tuple) and len(out) == 4:
            obs_dict, _, _, _ = out
        elif isinstance(out, tuple) and len(out) == 5:
            obs_dict = env.get_observations()
        else:
            raise RuntimeError("unexpected env.step return format")


if __name__ == "__main__":
    args = get_args()
    play(args)
