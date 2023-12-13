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
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils import math, common_values
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                         FaceBallReward, TouchBallReward)
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy


class TouchReward(RewardFunction):

    def __init__(self):
        super().__init__()
        self.velocity_player_ball_reward = VelocityPlayerToBallReward()
        self.face_ball_reward = FaceBallReward()
        self.distance_reward = LiuDistancePlayerToBallReward()
        self.touch_ball_reward = TouchBallReward()
        self.cumulative_reward = 0

    def reset(self, initial_state: GameState):
        self.velocity_player_ball_reward.reset(initial_state)
        self.face_ball_reward.reset(initial_state)
        self.distance_reward.reset(initial_state)
        self.touch_ball_reward.reset(initial_state)
        self.cumulative_reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward_vpb = self.velocity_player_ball_reward.get_reward(player, state, previous_action)
        reward_fb = self.face_ball_reward.get_reward(player, state, previous_action)
        reward_d = self.distance_reward.get_reward(player, state, previous_action)

        # touch_ball_reward = TouchBallReward()
        # reward_tb = touch_ball_reward.get_reward(player, state, previous_action)

        # + 400 * reward_tb
        reward = (reward_fb + reward_vpb + 4 * reward_d) - 6
        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


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


class CustomTerminalCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.last_touch != -1

if __name__ == "__main__":


    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 7

    # timesteps = seconds * 15

    max_steps = int(
        round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    def get_match():
        return Match(
            reward_function=TouchReward(),
            obs_builder=CustomObsBuilderBluePerspective(),
            terminal_conditions=[TimeoutCondition(max_steps), CustomTerminalCondition()],
            # spawn_opponents=True,
            action_parser=DefaultAction(),
            state_setter=DefaultState(),
        )


    name = "HitBallTest-2"
    log_path = os.path.join("Training", "Logs", name)
    ppo_path = os.path.join("Training", "Saved Models", name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=2,
                                 wait_time=1)
    env = VecMonitor(env)
    model = PPO(policy="MlpPolicy", env=env, verbose=2)
    # model = PPO.load(ppo_path, env)
    model.set_logger(logger)
    model.learn(20_000)
    # model.save(ppo_path)
    env.close()

# print(evaluate_policy(model, env, n_eval_episodes=10))

""" MlpPolicy:
ActorCriticPolicy(
  (features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (pi_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (vf_features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (mlp_extractor): MlpExtractor(
    (policy_net): Sequential(
      (0): Linear(in_features=26, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=26, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=64, out_features=8, bias=True)
  (value_net): Linear(in_features=64, out_features=1, bias=True)
)
"""