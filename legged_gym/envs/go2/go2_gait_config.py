from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2WalkCfg(GO2RoughCfg):
    class commands(GO2RoughCfg.commands):
        class ranges:
            lin_vel_x = [0.1, 0.5]   
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [0.0, 0.0]
    
    class rewards(GO2RoughCfg.rewards):
        gait = "walk"
        gait_period = 0.8
        duty_factor_target = 0.7
        gait_contact_sigma = 0.25
        base_height_target = 0.25
        soft_dof_pos_limit = 0.85
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.1
            orientation = -0.
            torques = -0.0002
            dof_vel = -5e-7
            dof_acc = -0.02
            feet_air_time =  0.8
            collision = -1.5
            duty_factor = 1.0
            gait_contact = 1.0
            lateral_symmetry = -0.5   
            gait_phase = -1.0 

    class control( GO2RoughCfg.control ):
        control_type = 'P' 
        stiffness = {'joint': 18.0}
        damping = {'joint': 0.6}
        action_scale = 0.18
        decimation = 4


class GO2TrotCfg(GO2RoughCfg):
    class commands(GO2RoughCfg.commands):
        class ranges:
            lin_vel_x = [0.1, 1.5]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(GO2RoughCfg.rewards):
        gait = "trot"
        gait_period = 0.5
        duty_factor_target = 0.5
        gait_contact_sigma = 0.25
        base_height_target = 0.3
        soft_dof_pos_limit = 0.9
        class scales:
            tracking_lin_vel = 1.6
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.08
            orientation = -0.6
            torques = -0.00002
            dof_acc = -2.5e-7
            base_height = -1.0
            feet_air_time = 0.35
            collision = -1.2
            action_rate = -0.002
            duty_factor = 0.4
            gait_contact = 1.3
            gait_phase = -0.6
            lateral_symmetry = -0.2

class GO2PaceCfg(GO2RoughCfg):
    class commands(GO2RoughCfg.commands):
        class ranges:
            lin_vel_x = [0.5, 2.0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(GO2RoughCfg.rewards):
        gait = "pace"
        gait_period = 0.5
        duty_factor_target = 0.5
        gait_contact_sigma = 0.25
        base_height_target = 0.3
        soft_dof_pos_limit = 0.9
        class scales:
            tracking_lin_vel = 1.6
            tracking_ang_vel = 0.7
            lin_vel_z = -2.0
            ang_vel_xy = -0.08
            orientation = -0.8
            torques = -0.00002
            dof_acc = -2.5e-7
            base_height = -1.0
            feet_air_time = 0.3
            collision = -1.2
            action_rate = -0.002
            duty_factor = 0.4
            gait_contact = 1.3
            gait_phase = -0.6

class GO2BoundCfg(GO2RoughCfg):
    class commands(GO2RoughCfg.commands):
        class ranges:
            lin_vel_x = [0.8, 2.5]   
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(GO2RoughCfg.rewards):
        gait = "bound"
        gait_period = 0.45
        duty_factor_target = 0.45
        gait_contact_sigma = 0.25
        base_height_target = 0.32
        soft_dof_pos_limit = 0.9
        class scales:
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.6
            lin_vel_z = -1.5
            ang_vel_xy = -0.05
            orientation = -0.5
            torques = -0.000015
            dof_acc = -2.0e-7
            base_height = -0.8
            feet_air_time = 0.25
            collision = -1.2
            action_rate = -0.0015
            duty_factor = 0.3
            gait_contact = 1.5
            gait_phase = -0.5
            lateral_symmetry = -0.15

class GO2WalkCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_walk'
        run_name = 'walk_gait'

class GO2TrotCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_trot'
        run_name = 'trot_gait'

class GO2PaceCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_pace'
        run_name = 'pace_gait'

class GO2BoundCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = 'go2_bound'
        run_name = 'bound_gait'
