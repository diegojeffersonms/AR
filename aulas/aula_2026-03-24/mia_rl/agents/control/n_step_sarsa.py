from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the n-step bootstrap.

        1. With probability `self.epsilon`, choose a random action from `self.actions`.
        2. Otherwise choose an action with the highest current action-value.
        3. Store the chosen action in `self._selected_actions[state]` and return it.
        """
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            best_value = max(self.action_value_of(state, action) for action in self.actions)
            best_actions = [action for action in self.actions if self.action_value_of(state, action) == best_value]
            action = self.rng.choice(best_actions)

        self._selected_actions[state] = action
        return action

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Store the transition and update the oldest state-action when possible.

        1. Append each transition to `self._pending_transitions`.
        2. If the episode ended, keep updating and removing the oldest transition until the buffer is empty.
        3. Otherwise, once the buffer length reaches `self.n_steps`, update the oldest transition and remove it.
        4. Reuse `_update_oldest_transition()` for the actual target computation.
        """
        self._pending_transitions.append(transition)

        if transition.done:
            while self._pending_transitions:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)

        elif len(self._pending_transitions) >= self.n_steps:
            self._update_oldest_transition()
            self._pending_transitions.pop(0)

    def _update_oldest_transition(self) -> None:
        """Compute the n-step Sarsa target for the oldest transition in the buffer.

        1. Build a window with at most `self.n_steps` transitions starting from the oldest one.
        2. Sum the discounted rewards inside that window.
        3. If the window has exactly `self.n_steps` transitions and is non-terminal, bootstrap from
            `Q(last_step.next_state, cached_next_action)`.
        4. Apply the incremental update with `self.alpha` to the oldest `(state, action)` pair.
        """
        window = self._pending_transitions[:self.n_steps]
        first_transition = window[0]
        state, action = first_transition.state, first_transition.action
        rewards_sum = sum(transition.reward * (self.gamma ** i) for i, transition in enumerate(window))

        if len(window) == self.n_steps and not window[-1].done:
            last_transition = window[-1]
            next_state = last_transition.next_state
            cached_next_action = self._selected_actions[next_state]
            rewards_sum += (self.gamma ** self.n_steps) * self.action_value_of(next_state, cached_next_action)

        self.Q[(state, action)] += self.alpha * (rewards_sum - self.action_value_of(state, action))

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])
    
    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda action: self.action_value_of(state, action))
