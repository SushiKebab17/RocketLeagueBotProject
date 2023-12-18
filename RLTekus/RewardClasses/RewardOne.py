import numpy as np
from rlgym_sim.utils import common_values
from rlgym_sim.utils.reward_functions import RewardFunction, DefaultReward
from rlgym_sim.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                             FaceBallReward, TouchBallReward)
from rlgym_sim.utils.gamestates import GameState, PlayerData

"""
    This Reward Class is for making Tekus hit the ball from time to time.
    Part 1.1 in the Objectives.
"""


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
        reward_fb = self.face_ball_reward.get_reward(player, state, previous_action)
        reward_tb = self.touch_ball_reward.get_reward(player, state, previous_action)
        reward_vpb = self.velocity_player_to_ball_reward.get_reward(player, state, previous_action)
        reward_d = self.distance_to_ball_reward.get_reward(player, state, previous_action)

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
