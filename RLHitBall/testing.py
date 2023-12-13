import rlgym_sim as rlgym
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.vec_env import VecMonitor
import os
from rlgym.envs import Match
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils import math, common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                         FaceBallReward, TouchBallReward)
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from main import TouchReward, CustomTerminalCondition, CustomObsBuilderBluePerspective

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 7

max_steps = int(
    round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

env = rlgym.make(reward_fn=TouchReward(), obs_builder=CustomObsBuilderBluePerspective(),
                     terminal_conditions=[TimeoutCondition(max_steps), CustomTerminalCondition()],
                     spawn_opponents=False)

name = "HitBallTest1-DEFAULT"
log_path = os.path.join("Training", "Logs", name)
ppo_path = os.path.join("Training", "Saved Models", name)
logger = configure(log_path, ["stdout", "csv", "tensorboard"])

env = SB3SingleInstanceEnv(env)
env = VecMonitor(env) #logs mean reward and ep length
# env = VecNormalize(env, norm_obs=False, gamma=gamma) # normalises rewards

learner = PPO(policy="MlpPolicy", env=env, verbose=1)
learner.set_logger(logger)
learner.learn(10_000)

# model = PPO.load(ppo_path, env)

# scores = []
# episodes = 5
# for episode in range(1, episodes + 1):
#     obs = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action, _ = model.predict(obs)  # env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         score += reward
#     print("Episode:{} Score:{}".format(episode, score))
#     scores.append(score)
# print(sum(scores)/len(scores))
