# import rlgym_sim as rlgym
import rlgym
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import os
from rlgym.envs import Match
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils import math, common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                         FaceBallReward, TouchBallReward, AlignBallGoal, LiuDistanceBallToGoalReward)
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from main import TouchReward, CustomTerminalCondition, CustomObsBuilderBluePerspective

import torch.nn as nn

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 30


class Two(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.touch_ball_reward = TouchBallReward()
        self.velocity_player_to_ball_reward = VelocityPlayerToBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()
        self.distance_to_goal_reward = LiuDistanceBallToGoalReward()
        self.orange_score = 0
        self.blue_score = 0
        self.elapsed = 0

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.touch_ball_reward.reset(initial_state)
        self.velocity_player_to_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)
        self.distance_to_goal_reward.reset(initial_state)
        self.orange_score = initial_state.orange_score
        self.blue_score = initial_state.blue_score
        self.elapsed = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        self.elapsed += 1

        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        reward_tb = self.touch_ball_reward.get_reward(
            player, state, previous_action)
        reward_vpb = self.velocity_player_to_ball_reward.get_reward(
            player, state, previous_action
        )
        reward_d = self.distance_to_ball_reward.get_reward(
            player, state, previous_action
        )
        reward_dg = self.distance_to_goal_reward.get_reward(
            player, state, previous_action
        )

        # array of these
        reward_types = np.array(
            [reward_fb, reward_tb, reward_vpb, reward_d, reward_dg])
        # array of weights
        reward_weights = np.array([1, 6, 3, 2, 5])

        # return dot
        reward = np.dot(reward_types, reward_weights)
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        buff = 300 * (30 * 30 - self.elapsed)  # 2500
        if player.team_num == common_values.ORANGE_TEAM:
            if state.orange_score > self.orange_score:
                reward += buff
            elif state.blue_score > self.blue_score:
                reward -= buff
        else:
            if state.orange_score > self.orange_score:
                reward -= buff
            elif state.blue_score > self.blue_score:
                reward += buff
        return reward


max_steps = int(
    round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

env = rlgym.make(game_speed=1,
                 reward_fn=Two(),
                 obs_builder=AdvancedObs(),
                 terminal_conditions=[TimeoutCondition(
                     max_steps), GoalScoredCondition()],
                 action_parser=KBMAction(),
                 state_setter=DefaultState(),
                 team_size=1,
                 spawn_opponents=True)


# def get_match():
#     return Match(
#         reward_function=TouchReward(),
#         obs_builder=AdvancedObs(),
#         terminal_conditions=[TimeoutCondition(
#             max_steps), GoalScoredCondition()],
#         action_parser=DefaultAction(),
#         state_setter=DefaultState(),
#         team_size=1,
#         spawn_opponents=True,
#     )

# "TekusReward1-1E-FINAL2"
name = "TekusReward1-1E-ALONEcont2"
# log_path = os.path.join("Training", "Logs", name)
ppo_path = os.path.join("Training", "Saved Models", name)
# logger = configure(log_path, ["stdout", "csv", "tensorboard"])

env = SB3SingleInstanceEnv(env)
env = VecMonitor(env)  # logs mean reward and ep length
# env = VecNormalize(env, norm_obs=False, gamma=0.99)  # normalises rewards

# learner = PPO(policy="MlpPolicy", env=env, verbose=1)
# learner.set_logger(logger)
# learner.learn(10_000)
# print(learner.policy)
# learner.save(ppo_path)

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


# model = PPO.load(ppo_path, env)
model = PPO.load(ppo_path, custom_objects=dict(
    policy_kwargs=dict(
        activation_fn=lambda: nn.GELU("tanh"),
        net_arch=(dict(pi=[256, 64, 128], vf=[256, 64, 128])),
    )
)
)

print(model.policy)
# model.learn(10_000)
# model.save(ppo_path)

scores = []
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    # print(obs.shape)  # type: ignorehh
    done = [False, False]
    score = 0

    while not any(done):
        # env.action_space.sample()
        # time.sleep(1)
        action, _ = model.predict(obs)  # type: ignore
        # print(action)
        # print()
        # action[:, 0:2] = action[:, 0:2] - 1
        obs, reward, done, info = env.step(action)
        # print(done)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
    scores.append(score)
print(scores)
print(sum(scores)/len(scores))
