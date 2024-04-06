
from rlgym_sim.utils.terminal_conditions import TerminalCondition
from rlgym_compat import GameState
# from rlgym_sim.utils.gamestates import GameState


class GoalScoredCondition(TerminalCondition):
    """
    A condition that will terminate an episode as soon as a goal is scored by either side.
    """

    def __init__(self):
        super().__init__()
        self.blue_score = 0
        self.orange_score = 0

    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        """
        Check to see if the game score for either team has been changed. If either score has changed, update the current
        known scores for both teams and return `True`. Note that the known game scores are never reset for this object
        because the game score is not set to 0 for both teams at the beginning of an episode.
        """

        if current_state.blue_score != self.blue_score or current_state.orange_score != self.orange_score:
            self.blue_score = current_state.blue_score
            self.orange_score = current_state.orange_score
            # print("GOAL",current_state.blue_score, current_state.orange_score, self.blue_score, self.orange_score)
            return True
        return False
