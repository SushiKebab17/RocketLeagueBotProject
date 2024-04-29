import rlgym_sim as rlgym
import gym
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
import os
from rlgym_sim.envs import Match
from RewardClasses import RewardOne
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
)
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils.obs_builders import AdvancedObs

from sb3_multiple_instance_env import SB3MultipleInstanceEnv
from kbm_act import KBMAction

import torch.nn as nn

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    print(datetime.datetime.now())
    # ----- RLGym_Sim parameters -----
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 30
    max_steps = int(
        round(ep_len_seconds * physics_ticks_per_second / default_tick_skip)
    )  # timesteps = seconds * 15

    instances_num = 30

    # ----- SB3 PPO hyperparameters -----
    learning_rate = 0.0002
    gamma = 0.97
    gae_lambda = 0.91
    # activation is GELU with the Tanh approximation
    policy_kwargs = dict(
        activation_fn=lambda: nn.GELU("tanh"),  # lambda: nn.GELU("tanh")
        net_arch=(dict(pi=[256, 64, 128], vf=[256, 64, 128])),  # 64x64
    )

    # ----- SB3 Training parameters -----
    n_steps = 2048
    batch_size = 64
    n_epochs = 10

    def get_match():
        return Match(
            reward_function=RewardOne.OneD(),
            obs_builder=AdvancedObs(),
            terminal_conditions=[TimeoutCondition(max_steps), GoalScoredCondition()],
            action_parser=KBMAction(),
            state_setter=DefaultState(),
            team_size=1,
            spawn_opponents=True,
        )

    name = "TekusReward1-1E-FINAL3cont"
    name_save = "TekusReward1-1Dalone"
    log_path = os.path.join("Training", "Logs", name_save)
    ppo_path_load = os.path.join("Training", "Saved Models", name)
    ppo_path_save = os.path.join("Training", "Saved Models", name_save)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, num_instances=instances_num, wait_time=1
    )

    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=2,
        policy_kwargs=policy_kwargs,
        device="cpu",
        stats_window_size=instances_num,
    )

    # print("loading: " + name)
    # model = PPO.load(ppo_path_load, env)

    model.set_logger(logger)
    total_timesteps = 68_500_000
    print("saving: " + name_save)
    print(total_timesteps)
    model.learn(total_timesteps)
    model.save(ppo_path_save)
    env.close()
