from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

from legged_gym.envs.go2.go2_gait_config import (
    GO2WalkCfg, GO2WalkCfgPPO,
    GO2TrotCfg, GO2TrotCfgPPO, 
    GO2PaceCfg, GO2PaceCfgPPO,
    GO2BoundCfg, GO2BoundCfgPPO
)

from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot
from .go2.go2_env import LeggedRobotR2
from .go2.go2_flat_gait_env import Go2FlatGaitRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())

task_registry.register("go2_walk", Go2FlatGaitRobot, GO2WalkCfg(), GO2WalkCfgPPO())
task_registry.register("go2_trot", Go2FlatGaitRobot, GO2TrotCfg(), GO2TrotCfgPPO())
task_registry.register("go2_pace", Go2FlatGaitRobot, GO2PaceCfg(), GO2PaceCfgPPO())
task_registry.register("go2_bound", Go2FlatGaitRobot, GO2BoundCfg(), GO2BoundCfgPPO())
task_registry.register("trot", Go2FlatGaitRobot, GO2TrotCfg(), GO2TrotCfgPPO())
task_registry.register("pace", Go2FlatGaitRobot, GO2PaceCfg(), GO2PaceCfgPPO())
task_registry.register("bound", Go2FlatGaitRobot, GO2BoundCfg(), GO2BoundCfgPPO())

task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
