import rlgym_sim as rlgym
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import os
from rlgym_sim.envs import (Match)
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition, GoalScoredCondition
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils import math, common_values
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                             FaceBallReward, TouchBallReward)
# from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
# from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from sb3_multiple_instance_env import SB3MultipleInstanceEnv
from kbm_act import KBMAction
from GELUTanh import GELUTanh

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn


class One(RewardFunction):

    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.touch_ball_reward = TouchBallReward()
        self.velocity_player_to_ball_reward = VelocityPlayerToBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()
        self.orange_score = 0
        self.blue_score = 0

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.touch_ball_reward.reset(initial_state)
        self.velocity_player_to_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)
        self.orange_score = initial_state.orange_score
        self.blue_score = initial_state.blue_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        reward_tb = self.touch_ball_reward.get_reward(
            player, state, previous_action)
        reward_vpb = self.velocity_player_to_ball_reward.get_reward(
            player, state, previous_action)
        reward_d = self.distance_to_ball_reward.get_reward(
            player, state, previous_action)

        reward = (reward_fb * 1 +
                  reward_tb * 6 +
                  reward_vpb * 3 +
                  reward_d * 3)
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = self.get_reward(player, state, previous_action)
        buff = 500
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


class CustomObsBuilderBluePerspective(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        obs = []

        # If this observation is being built for a player on the orange team,
        # we need to invert all the physics data we use.
        inverted = player.team_num == common_values.ORANGE_TEAM

        if inverted:
            obs += state.inverted_ball.serialize()
        else:
            obs += state.ball.serialize()

        for player in state.players:
            if inverted:
                obs += player.inverted_car_data.serialize()
            else:
                obs += player.car_data.serialize()

        return np.asarray(obs, dtype=np.float32)


if __name__ == "__main__":

    # &&& RLGym parameters &&&
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 15
    max_steps = int(round(ep_len_seconds * physics_ticks_per_second /
                    default_tick_skip))  # timesteps = seconds * 15

    policy_kwargs = dict(
        activation_fn=GELUTanh,
        net_arch=(
            dict(
                pi=[512, 512, 64],
                vf=[512, 512, 64]
            )
        )
    )

    def get_match():
        return Match(
            reward_function=One(),
            obs_builder=CustomObsBuilderBluePerspective(),
            terminal_conditions=[TimeoutCondition(
                max_steps), GoalScoredCondition()],
            spawn_opponents=True,
            action_parser=KBMAction(),
            state_setter=DefaultState(),
        )

    name = "HitBallTest-2"
    log_path = os.path.join("Training", "Logs", name)
    ppo_path = os.path.join("Training", "Saved Models", name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=1,
                                 wait_time=1)
    env = VecMonitor(env)
    model = PPO(policy="MlpPolicy", env=env,
                verbose=2, policy_kwargs=policy_kwargs)
    # model = PPO.load(ppo_path, env)
    model.set_logger(logger)
    model.learn(40_000)
    # model.save(ppo_path)
    env.close()

# print(evaluate_policy(model, env, n_eval_episodes=10))

""" 
print(model.policy) represents this:
26 -> 64 -> 64 -> 8
"""
