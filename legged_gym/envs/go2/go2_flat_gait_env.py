import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2FlatGaitRobot(LeggedRobot):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.gait_period = getattr(self.cfg.rewards, "gait_period", self.gait_period)

    def _init_buffers(self):
        super()._init_buffers()
        self._configure_gait_phase()

    def _configure_gait_phase(self):
        gait = getattr(self.cfg.rewards, "gait", "walk")
        phase_offsets = {
            "walk": {
                self.fl_idx: 0.0,
                self.fr_idx: 0.25,
                self.rr_idx: 0.5,
                self.rl_idx: 0.75,
            },
            "trot": {
                self.fl_idx: 0.0,
                self.rr_idx: 0.0,
                self.fr_idx: 0.5,
                self.rl_idx: 0.5,
            },
            "pace": {
                self.fl_idx: 0.0,
                self.rl_idx: 0.0,
                self.fr_idx: 0.5,
                self.rr_idx: 0.5,
            },
            "bound": {
                self.fl_idx: 0.0,
                self.fr_idx: 0.0,
                self.rl_idx: 0.5,
                self.rr_idx: 0.5,
            },
        }
        if gait not in phase_offsets:
            raise ValueError(f"Unsupported gait '{gait}'. Expected one of {list(phase_offsets.keys())}.")

        self.phi[:] = 0.0
        for leg_idx, phase in phase_offsets[gait].items():
            self.phi[leg_idx] = phase
        self.r_des = getattr(self.cfg.rewards, "duty_factor_target", self.r_des)

    def _reward_gait_contact(self):
        desired = self._desired_contact_pattern()
        actual = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        mismatch = torch.abs(actual - desired).mean(dim=1)
        sigma = getattr(self.cfg.rewards, "gait_contact_sigma", 0.25)
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        return torch.exp(-mismatch / sigma) * moving
