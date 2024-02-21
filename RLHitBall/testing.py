import rlgym_sim as rlgym
# import rlgym
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.logger import CSVOutputFormat, configure
from stable_baselines3.common.vec_env import VecMonitor
import os
from rlgym_sim.envs import Match
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils import math, common_values
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions.common_rewards import (LiuDistancePlayerToBallReward, VelocityPlayerToBallReward,
                                                             FaceBallReward, TouchBallReward)
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
from sb3_multiple_instance_env import SB3MultipleInstanceEnv
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from main import TouchReward, CustomTerminalCondition, CustomObsBuilderBluePerspective


import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 30


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


max_steps = int(
    round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

env = rlgym.make(reward_fn=One(),
                 obs_builder=AdvancedObs(),
                 terminal_conditions=[TimeoutCondition(
                     max_steps), GoalScoredCondition()],
                 action_parser=KBMAction(),
                 state_setter=DefaultState(),
                 team_size=1,
                 spawn_opponents=True)

name = "TekusRLGYMSIMTest"
log_path = os.path.join("Training", "Logs", name)
ppo_path = os.path.join("Training", "Saved Models", name)
logger = configure(log_path, ["stdout", "csv", "tensorboard"])


def get_match():
    return Match(
        reward_function=TouchReward(),
        obs_builder=AdvancedObs(),
        terminal_conditions=[TimeoutCondition(
            max_steps), GoalScoredCondition()],
        action_parser=DefaultAction(),
        state_setter=DefaultState(),
        team_size=1,
        spawn_opponents=True,
    )


# if __name__ == "__main__":
    # env = SB3SingleInstanceEnv(env)
    # env = SB3MultipleInstanceEnv(
    # match_func_or_matches=get_match, num_instances=4, wait_time=1
    # )
    # env = VecMonitor(env)  # logs mean reward and ep length
    # env = VecNormalize(env, norm_obs=False, gamma=gamma) # normalises rewards

    # learner = PPO(policy="MlpPolicy", env=env, verbose=1)
    # learner.set_logger(logger)
    # learner.learn(10_000)
    # print(learner.policy)
    # learner.save(ppo_path)


model = PPO.load(ppo_path, env)

env = rlgym.make(
    spawn_opponents=True,
    reward_function=TouchReward(),
    obs_builder=AdvancedObs(),
    terminal_conditions=[TimeoutCondition(
        max_steps), GoalScoredCondition()],
    action_parser=KBMAction(),
    state_setter=DefaultState(),
    team_size=1,)

# scores = []
# episodes = 5
# for episode in range(1, episodes + 1):
#     obs = env.reset()
#     print(obs.shape)  # type: ignore
#     done = False
#     score = 0

#     while not done:
#         # env.action_space.sample()
#         action, _ = model.predict(obs)  # type: ignore
#         obs, reward, done, info = env.step(action)
#         score += reward
#     print("Episode:{} Score:{}".format(episode, score))
#     scores.append(score)
# print(sum(scores)/len(scores))
