from __future__ import annotations

from typing import Optional

import numpy as np

from mia_rl.core.base import Environment

KBanditState = int
KBanditAction = int


class KArmedBanditEnv(Environment[KBanditState, KBanditAction]):
	"""Multi-armed bandit environment.

	The environment is a continuing task with a single dummy state (0).
	At each step, the agent chooses an arm and receives a stochastic reward:

		R_t ~ N(q_*(A_t), 1)

	where q_* is the true action-value vector.
	"""

	def __init__(
		self,
		k: int = 10,
		stationary: bool = True,
		walk_std: float = 0.01,
		seed: Optional[int] = None,
	) -> None:
		if k <= 0:
			raise ValueError("k must be a positive integer.")
		if walk_std < 0:
			raise ValueError("walk_std must be non-negative.")

		self.k = k
		self.stationary = stationary
		self.walk_std = walk_std
		self.rng = np.random.default_rng(seed)

		self.q_true = np.zeros(self.k, dtype=float)
		self.optimal_action = 0
		self._state: KBanditState = 0

		self.reset()

	def reset(self) -> KBanditState:
		self.q_true = self.rng.normal(loc=0.0, scale=1.0, size=self.k)
		self.optimal_action = int(np.argmax(self.q_true))
		self._state = 0
		return self._state

	def available_actions(self, state: KBanditState) -> list[KBanditAction]:
		return list(range(self.k))

	def step(self, action: KBanditAction) -> tuple[KBanditState, float, bool]:
		if action < 0 or action >= self.k:
			raise ValueError(f"Invalid action: {action}. Expected an integer in [0, {self.k - 1}].")

		reward = float(self.rng.normal(loc=self.q_true[action], scale=1.0))

		if not self.stationary:
			self.q_true += self.rng.normal(loc=0.0, scale=self.walk_std, size=self.k)
			self.optimal_action = int(np.argmax(self.q_true))

		return self._state, reward, False

	def true_action_values(self) -> np.ndarray:
		return self.q_true.copy()
