import torch
from legged_gym.envs.base.legged_robot import LeggedRobot

class GO2MorAL(LeggedRobot):
    """仅在 LeggedRobot 上补 MorAL 接口：obs_dict keys + history + privileged_info(224)"""

    PRIV_DIM = 224

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # --- proprio history (proprio_hist) ---
        self.H = getattr(self.cfg.env, "num_histroy_obs", 4)
        obs_dim = self.cfg.env.num_observations
        self.proprio_hist = torch.zeros(self.num_envs, obs_dim * self.H, device=self.device)

        # --- privileged_info (224) ---
        self.privileged_info = torch.zeros(self.num_envs, self.PRIV_DIM, device=self.device)

        # --- morph true (9 dims) ---
        self.morph9 = torch.zeros(self.num_envs, 9, device=self.device)

    # MorAL runner 调用
    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self.compute_observations()
        self.proprio_hist[:] = 0.0
        self._update_proprio_hist()
        self._update_privileged_info()
        return self.get_observations()

    def get_observations(self):
        return {
            "obs": self.obs_buf,
            "privileged_info": self.privileged_info,
            "proprio_hist": self.proprio_hist,
        }

    # MorAL runner 期望 step 返回 (obs_dict, rewards, dones, infos)
    def step(self, actions):
        _, _, rew, done, infos = super().step(actions)

        self._update_proprio_hist()
        self._update_privileged_info()

        infos = infos if isinstance(infos, dict) else {}
        # 可选：如果你想在 infos 里也保留真值
        infos["v_true"] = self.base_lin_vel.detach()
        infos["mor_true_9"] = self.morph9.detach()

        return self.get_observations(), rew, done, infos

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.proprio_hist[env_ids] = 0.0
        self.privileged_info[env_ids] = 0.0

    # ---------------- helpers ----------------
    def _update_proprio_hist(self):
        obs = self.obs_buf
        obs_dim = obs.shape[1]
        target_dim = obs_dim * self.H
        if self.proprio_hist.shape[1] != target_dim:
            self.proprio_hist = torch.zeros(self.num_envs, target_dim, device=self.device)

        self.proprio_hist = torch.roll(self.proprio_hist, shifts=-obs_dim, dims=1)
        self.proprio_hist[:, -obs_dim:] = obs

    def _update_privileged_info(self):
        """
        填充 privileged_info[0:224]
        先保证维度与切片合法；信息可以逐步完善。
        """
        pi = self.privileged_info
        pi.zero_()

        # 0:3 real vel (确保在同一 device)
        pi[:, 0:3] = self.base_lin_vel.to(self.device)

        # 3:198 heights (195) ——没开 measure_heights 就保持 0
        # if hasattr(self, "measured_heights"):
        #     pi[:, 3:198] = self.measured_heights.to(self.device)

        # 198:200 push（2维）——没有就保持 0
        # if hasattr(self, "last_push_xy"):
        #     pi[:, 198:200] = self.last_push_xy.to(self.device)

        # 200:209 morph（9维）
        # 你现在的 self.morph9 很可能没初始化/全 0 -> 先用常数把分布拉回训练范围
        if hasattr(self, "morph9") and self.morph9 is not None:
            m = self.morph9
            # 支持 m 是 [9] 或 [N,9]
            if m.ndim == 1:
                m = m.view(1, 9).repeat(self.num_envs, 1)
            pi[:, 200:209] = m.to(self.device)
        else:
            # 最小可行：不要全 0，先用 0.5 占位（训练时常是归一化到 0~1）
            pi[:, 200:209] = 0.5

        # 209:223 kp_kd（14维）——先用 0.5 占位比全 0 更稳（避免分布漂移）
        # 如果你训练时这里是归一化后的 kp/kd，这里也应当给个中值
        pi[:, 209:223] = 0.5

        # 223:224 friction（1维）——强制写成 [N,1]
        if hasattr(self, "friction_coeffs") and self.friction_coeffs is not None:
            fr = self.friction_coeffs
            # 兼容 [N,1,1] / [N,1] / [N]
            fr = fr.view(self.num_envs, -1)[:, :1].to(self.device)
            pi[:, 223:224] = fr
        else:
            # 没有 friction_coeffs 就不要给 0，给 0.5（中值）先保证能走
            pi[:, 223:224] = 0.5
