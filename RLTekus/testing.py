import rlgym_sim as rlgym
import gym
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
import os
from rlgym_sim.envs import Match
from RewardClasses import RewardOne
from CustomObservation import CustomObservation
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

# This is a Test file used to test out the rlgym-sim environment, as well as hyperparameters.

if __name__ == "__main__":
    print(datetime.datetime.now())
    # ----- RLGym_Sim parameters -----
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 3
    max_steps = int(
        round(ep_len_seconds * physics_ticks_per_second / default_tick_skip)
    )
    # timesteps = seconds * 15

    instances_num = 30

    # ----- SB3 PPO hyperparameters -----
    learning_rate = 0.0003
    gamma = 0.99
    gae_lambda = 0.95
    # activation is GELU with the Tanh approximation
    policy_kwargs = dict(
        activation_fn=nn.ReLU,  # lambda: nn.GELU("tanh")
        net_arch=(dict(pi=[64, 64], vf=[64, 64])),
    )

    # ----- SB3 Training parameters -----
    n_steps = 2048
    batch_size = 64
    n_epochs = 10

    # function that returns the Match object, used when creating the environment
    def get_match():
        return Match(
            reward_function=RewardOne.One(),
            obs_builder=AdvancedObs(),
            terminal_conditions=[TimeoutCondition(
                max_steps), GoalScoredCondition()],
            action_parser=KBMAction(),
            state_setter=DefaultState(),
            team_size=1,
            spawn_opponents=True,
        )

    # initialising variables to do with saving the logs, and saving and loading the model
    name = "TekusTest"
    log_path = os.path.join("Training", "Logs", name)
    ppo_path = os.path.join("Training", "Saved Models", name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # create the environement, and wrap with VecMonitor and VecNormalize
    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, num_instances=instances_num, wait_time=1
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    # define the model being trained by the PPO object
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
    )

    # set the logger to the model, set the training time, train, then save, then close the environment
    model.set_logger(logger)
    # model.learn(35_000_000)
    print(model.policy)
    # model.save(ppo_path)
    env.close()
