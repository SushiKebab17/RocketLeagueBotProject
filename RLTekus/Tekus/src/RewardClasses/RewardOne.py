import numpy as np
from rlgym_sim.utils import common_values
from rlgym_sim.utils.reward_functions import RewardFunction, DefaultReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    LiuDistancePlayerToBallReward,
    VelocityPlayerToBallReward,
    FaceBallReward,
    TouchBallReward,
    LiuDistanceBallToGoalReward,
    AlignBallGoal,
    VelocityBallToGoalReward,
)
from rlgym_sim.utils.gamestates import GameState, PlayerData

"""
    These Reward Class are for making Tekus hit the ball from time to time.
    Part 1.1 in the Objectives.
"""

"""
    OneA - rewards just facing the ball
"""


class OneA(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        return reward_fb

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        return reward


class OneB(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        reward_d = self.distance_to_ball_reward.get_reward(
            player, state, previous_action
        )

        reward = reward_fb * 1 + reward_d * 3
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        return reward


class OneC(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.velocity_player_to_ball_reward = VelocityPlayerToBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.velocity_player_to_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward_fb = self.face_ball_reward.get_reward(
            player, state, previous_action)
        reward_vpb = self.velocity_player_to_ball_reward.get_reward(
            player, state, previous_action
        )
        reward_d = self.distance_to_ball_reward.get_reward(
            player, state, previous_action
        )

        reward = reward_fb * 1 + reward_vpb * 3 + reward_d * 3
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        return reward


class OneD(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.touch_ball_reward = TouchBallReward()
        self.velocity_player_to_ball_reward = VelocityPlayerToBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.touch_ball_reward.reset(initial_state)
        self.velocity_player_to_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
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

        reward = reward_fb * 1 + reward_tb * 6 + reward_vpb * 3 + reward_d * 3
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        return reward


class OneE(RewardFunction):
    def __init__(self):
        self.face_ball_reward = FaceBallReward()
        self.touch_ball_reward = TouchBallReward()
        self.velocity_player_to_ball_reward = VelocityPlayerToBallReward()
        self.distance_to_ball_reward = LiuDistancePlayerToBallReward()
        self.orange_score = 0
        self.blue_score = 0
        self.elapsed = 0

    def reset(self, initial_state: GameState):
        self.face_ball_reward.reset(initial_state)
        self.touch_ball_reward.reset(initial_state)
        self.velocity_player_to_ball_reward.reset(initial_state)
        self.distance_to_ball_reward.reset(initial_state)
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

        reward = reward_fb * 1 + reward_tb * 6 + reward_vpb * 3 + reward_d * 3
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        buff = 2500  # 300 * (30 * 30 - self.elapsed)
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


class TwoA(RewardFunction):
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
        reward_weights = np.array([1, 6, 3, 3, 3])

        # return dot
        reward = np.dot(reward_types, reward_weights)
        return reward

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.get_reward(player, state, previous_action)
        buff = 2500
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
