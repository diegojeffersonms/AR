from __future__ import annotations

from mia_rl.core.base import Policy
from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeState


class ThresholdPolicy(Policy[TicTacToeState, TicTacToeAction]):
    def __init__(self, threshold: int = 20):
        self.threshold = threshold

    def select_action(self, state: TicTacToeState) -> TicTacToeAction:
        player_sum, _, _ = state
        return "hit" if player_sum < self.threshold else "stick"
