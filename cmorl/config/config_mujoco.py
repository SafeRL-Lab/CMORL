from cmorl.environments.mujoco.half_cheetah_v4 import HalfCheetahEnv
from cmorl.environments.mujoco.half_cheetah_v4_soft_v1 import HalfCheetah_Soft_v1_Env
from cmorl.environments.mujoco.humanoid_v4 import HumanoidEnv
from cmorl.environments.mujoco.walker2d_v4 import Walker2dEnv
from cmorl.environments.mujoco.ant_v4 import AntEnv
from cmorl.environments.mujoco.hopper_v4 import HopperEnv
from cmorl.environments.mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from cmorl.environments.mujoco.pusher_v4 import PusherEnv
from cmorl.environments.mujoco.reacher_v4 import ReacherEnv
from cmorl.environments.mujoco.swimmer_v4 import SwimmerEnv

# dm_control
from dm_control import suite
from dm_control import viewer
import numpy as np

def get_env_mujoco_config(args):
    # env = gym.make(args.env_name)
    # todo: create new environments
    if args.env_name == "HalfCheetah-v4":
        env = HalfCheetahEnv(goal_vel=0.3)
        print("HalfCheetah-v4 Environments")
    elif args.env_name == "Humanoid-v4":
        env = HumanoidEnv()
        print("Humanoid-v4 Environments")
    elif args.env_name == "Walker-v4":
        env = Walker2dEnv()
        print("Walker-v4 Environments")
    elif args.env_name == "same_Ant-v4":
        env = AntEnv()
        print("same_Ant-v4 Environments")
    elif args.env_name == "Hopper-v4":
        env = HopperEnv()
        print("Hopper-v4 Environments")
    elif args.env_name == "HumanoidStandup-v4":
        env = HumanoidStandupEnv()
        print("HumanoidStandup-v4 Environments")
    elif args.env_name == "Pusher-v4":
        env = PusherEnv()
        print("Pusher-v4 Environments")
    elif args.env_name == "Swimmer-v4":
        env = SwimmerEnv()
        print("Swimmer-v4 Environments")
    elif args.env_name == "HalfCheetah-v4-Soft-v1":
        env = HalfCheetah_Soft_v1_Env(goal_vel=0.3)
        print("HalfCheetah-v4-Soft-v1 Environments")
    elif args.env_name == "Reacher-v4":  # need to revise
        env = ReacherEnv()
        print("Reacher-v4 Environments")
        # todo: dm_control environments start
    elif args.env_name == "Humanoid-dm":
        env = suite.load(domain_name="humanoid", task_name="walk")  # stand, walk, run
        print("Humanoid-dm Environments")
    elif args.env_name == "Walker-dm":
        env = suite.load(domain_name="walker", task_name="walk") # stand, walk, run
        print("Walker-dm Environments")


    else:
        print("error! Please input a correct task's name!")
    return env
