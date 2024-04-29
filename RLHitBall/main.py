import rlgym
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import os
from rlgym.envs import (Match)
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils import math, common_values
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                         FaceBallReward, TouchBallReward)
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

# from sb3_multiple_instance_env import SB3MultipleInstanceEnv
# from kbm_act import KBMAction

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

# This is the file used to train HitBall in RLGym


# The TouchReward class used for HitBall
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
        reward_vpb = self.velocity_player_ball_reward.get_reward(
            player, state, previous_action)
        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        reward_d = self.distance_reward.get_reward(
            player, state, previous_action)

        # touch_ball_reward = TouchBallReward()
        # reward_tb = touch_ball_reward.get_reward(player, state, previous_action)

        # + 400 * reward_tb
        reward = (reward_fb + reward_vpb + 4 * reward_d) - 6
        self.cumulative_reward += reward
        return reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = self.get_reward(player, state, previous_action)
        return self.get_reward(player, state, previous_action)


# The custom observation builder used for HitBall
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


# Use the last touched terminal condition
class CustomTerminalCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        print("touched")
        return current_state.last_touch != -1


if __name__ == "__main__":

    # ----- RLGym parameters -----
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 7
    max_steps = int(round(ep_len_seconds * physics_ticks_per_second /
                    default_tick_skip))  # timesteps = seconds * 15

    # function that returns the Match object, used when creating the environment
    def get_match():
        return Match(
            reward_function=TouchReward(),
            obs_builder=CustomObsBuilderBluePerspective(),
            terminal_conditions=[TimeoutCondition(
                max_steps), BallTouchedCondition()],
            # spawn_opponents=True,
            action_parser=DefaultAction(),
            state_setter=DefaultState(),
        )

    # initialising variables to do with saving the logs, and saving and loading the model
    name = "HitBallTest-3"
    log_path = os.path.join("Training", "Logs", name)
    ppo_path = os.path.join("Training", "Saved Models", name)
    logger = configure(log_path, ["stdout", "csv", "tensorboard"])

    # create the environment, and wrap with VecMonitor
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match,
                                 num_instances=4,
                                 wait_time=1)
    env = VecMonitor(env)

    # create or load model
    model = PPO(policy="MlpPolicy", env=env, verbose=2)
    # model = PPO.load(ppo_path, env)

    # set logger, train, save to the path, and close the environment
    model.set_logger(logger)
    model.learn(20_000)
    # model.save(ppo_path)
    env.close()

# evaluate policy using the SB3 eval policy
# print(evaluate_policy(model, env, n_eval_episodes=10))
