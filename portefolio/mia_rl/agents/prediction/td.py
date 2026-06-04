from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        for transition in episode.transitions:
            bootstrap = 0.0 if transition.done or transition.next_state is None else self.V[transition.next_state]
            target = transition.reward + self.gamma * bootstrap
            self.V[transition.state] += self.alpha * (target - self.V[transition.state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])

class TDNPrediction(TD0Prediction):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0, n: int = 3):
        super().__init__(alpha=alpha, gamma=gamma)
        self.n = n

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        for i in range(len(episode.transitions)):
            G = 0.0
            for j in range(i, min(i + self.n, len(episode.transitions))):
                transition = episode.transitions[j]
                G += (self.gamma ** (j - i)) * transition.reward
                if transition.done or transition.next_state is None:
                    break
            if j < len(episode.transitions) and not episode.transitions[j].done and episode.transitions[j].next_state is not None:
                G += (self.gamma ** (j - i + 1)) * self.V[episode.transitions[j].next_state]
            self.V[episode.transitions[i].state] += self.alpha * (G - self.V[episode.transitions[i].state])
