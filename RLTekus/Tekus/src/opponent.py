from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
import math
import time

from action.kbm_act import KBMAction
from agent2 import Agent
from obs.CustomObservation import CustomObservation
from obs.advanced_obs import AdvancedObs
from obs.default_obs import DefaultObs
from RewardClasses.RewardOne import TwoA
# from rlgym_sim.utils.gamestates import GameState
from rlgym_compat import GameState
from terminal import GoalScoredCondition

# This class is for updating the controls for the Orange team agent.


class Tekus(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = AdvancedObs()
        self.act_parser = KBMAction()
        # neural network logic goes inside the Agent class, go take a look inside src/agent.py
        self.agent = Agent()
        self.tick_skip = 8

        self.name = name

        self.fp = open("data.txt", "a")
        self.prev_action = np.zeros(8)
        self.reward_function = TwoA()
        self.cumulative_reward_1 = 0
        self.cumulative_reward_2 = 0
        self.terminal_condition = GoalScoredCondition()

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.prev_tick = 0
        self.ticks_elapsed_since_update = 0
        self.ticks_since_tried_score = 0
        self.done = False
        self.started = False
        self.checked_kickoff = False
        self.prev_time_remaining = 300
        print('Tekus Ready - Index:', index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

        self.reward_function.reset(self.game_state)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # start = time.time()
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        # team_info_0 = packet.teams[0].score
        # team_info_1 = packet.teams[1].score
        # curr_time_remaining = math.ceil(packet.game_info.game_time_remaining)
        # if self.prev_time_remaining != curr_time_remaining:
        #     print(f"{curr_time_remaining}s {team_info_0}-{team_info_1}")
        #     self.prev_time_remaining = curr_time_remaining

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [
                p for p in self.game_state.players if p.team_num == self.team]
            opponents = [
                p for p in self.game_state.players if p.team_num != self.team]

            if len(opponents) == 0:
                # There's no opponent, we assume this model is 1v0
                self.game_state.players = [player]
            else:
                # Sort by distance to ball
                teammates.sort(key=lambda p: np.linalg.norm(
                    self.game_state.ball.position - p.car_data.position))
                opponents.sort(key=lambda p: np.linalg.norm(
                    self.game_state.ball.position - p.car_data.position))

                # Grab opponent in same "position" relative to it's teammates
                opponent = opponents[min(
                    teammates.index(player), len(opponents) - 1)]

                self.game_state.players = [player, opponent]

            obs = self.obs_builder.build_obs(
                player, self.game_state, self.action)
            self.action = self.act_parser.parse_actions(
                self.agent.act(obs), self.game_state)[0]  # Dim is (N, 8)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)
            # reward = self.reward_function.get_reward(
            #     player, self.game_state, self.prev_action
            # )
            # self.cumulative_reward += reward
            # self.fp.write(f"{self.cumulative_reward}\n")
            # self.prev_action = self.action

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        # print(time.time() - start)

        # CODE BELOW IS FOR OUTPUTTING BOT REWARD, credit to JPK314 on Discord for assisting me with the code below.

        # cur_tick = packet.game_info.frame_num
        # delta = cur_tick - self.prev_tick
        # self.prev_tick = cur_tick
        # self.ticks += delta
        # self.ticks_elapsed_since_update += delta
        # self.ticks_since_tried_score += delta

        # self.game_state.decode(packet, delta)
        # if packet.game_info.is_kickoff_pause and not self.checked_kickoff:
        #     self.checked_kickoff = True
        #     self.reward_function.reset(self.game_state)
        #     self.terminal_condition.reset(self.game_state)
        #     self.obs_builder.reset(self.game_state)
        #     self.ticks_elapsed_since_update = 0
        #     self.inverse_returns = 0
        #     self.done = False
        #     self.started = True

        # if not packet.game_info.is_kickoff_pause:
        #     self.checked_kickoff = False

        # if self.started:
        #     if self.ticks_elapsed_since_update >= self.tick_skip and not self.done:
        #         self.done = self.terminal_condition.is_terminal(
        #             self.game_state)

        #         player = self.game_state.players[self.index]
        #         teammates = [
        #             p for p in self.game_state.players if p.team_num == self.team]
        #         opponents = [
        #             p for p in self.game_state.players if p.team_num != self.team]

        #         if len(opponents) == 0:
        #             # There's no opponent, we assume this model is 1v0
        #             self.game_state.players = [player]
        #         else:
        #             # Sort by distance to ball
        #             teammates.sort(key=lambda p: np.linalg.norm(
        #                 self.game_state.ball.position - p.car_data.position))
        #             opponents.sort(key=lambda p: np.linalg.norm(
        #                 self.game_state.ball.position - p.car_data.position))

        #             # Grab opponent in same "position" relative to it's teammates
        #             opponent = opponents[min(
        #                 teammates.index(player), len(opponents) - 1)]

        #             self.game_state.players = [player, opponent]

        #         self.reward_function.pre_step(self.game_state)

        #         # Tekus' team
        #         if self.done:
        #             reward = self.reward_function.get_final_reward(
        #                 player, self.game_state, self.prev_action
        #             )
        #         else:
        #             reward = self.reward_function.get_reward(
        #                 player, self.game_state, self.prev_action
        #             )
        #         obs = self.obs_builder.build_obs(
        #             player, self.game_state, self.prev_action
        #         )
        #         # action_idx, _ = self.policy.get_action(obs)
        #         self.action = self.act_parser.parse_actions(
        #             self.agent.act(obs), self.game_state)[0]
        #         self.update_controls(self.action)
        #         self.prev_action = self.action
        #         self.ticks_elapsed_since_update = 0
        #         self.cumulative_reward_1 += reward
        #         # Write to file
        #         self.fp.write(f"{self.name}: {self.cumulative_reward_1}\n")
        #         if self.done:
        #             self.fp.write(f"DONE\n")
        #             self.cumulative_reward_1 = 0
        #         self.fp.flush()

        #         # Bot's team
        #         if self.done:
        #             reward = self.reward_function.get_final_reward(
        #                 opponents[0], self.game_state, self.prev_action
        #             )
        #         else:
        #             reward = self.reward_function.get_reward(
        #                 opponents[0], self.game_state, self.prev_action
        #             )
        #         obs = self.obs_builder.build_obs(
        #             opponents[0], self.game_state, self.prev_action
        #         )

        #         # action_idx, _ = self.policy.get_action(obs)
        #         # action = self.act_parser.parse_actions(
        #         #     self.agent.act(obs), self.game_state)[0]
        #         # self.update_controls(action)

        #         self.cumulative_reward_2 += reward
        #         # Write to file
        #         self.fp.write(f"Rookie: {self.cumulative_reward_2}\n")
        #         if self.done:
        #             self.fp.write(f"DONE\n")
        #             self.cumulative_reward_2 = 0
        #         self.fp.flush()

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
