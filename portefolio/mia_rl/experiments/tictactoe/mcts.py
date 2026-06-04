from __future__ import annotations

from mia_rl.agents.planning.mcts import MCTSAgent
from mia_rl.envs.tictactoe import (
    TicTacToeAction,
    TicTacToeEnv,
)
from mia_rl.policies.tictactoe import Policy, random_action


# ── Policy wrapper ────────────────────────────────────────────────────────────

def make_mcts_policy(agent: MCTSAgent) -> Policy:
    """Wrap an MCTSAgent as a ``Policy`` callable compatible with ``play_game``."""

    def policy(env: TicTacToeEnv) -> TicTacToeAction:
        return agent.select_action(env.board, env.current_player)

    return policy

# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_vs_policy(
    env: TicTacToeEnv,
    agent: MCTSAgent,
    policy_opponent: Policy,
    n_games: int = 200,
    as_player: int = 1,
) -> tuple[float, float, float]:
    """Evaluate MCTS win/draw/loss rates against a uniform-random opponent.

    Args:
        env:       TicTacToe environment instance.
        agent:     MCTSAgent to evaluate.
        n_games:   number of evaluation games.
        as_player: +1 → MCTS plays X (first mover); -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` — three fractions summing to 1.
    """
    wins = draws = losses = 0

    for _ in range(n_games):
        env.reset()
        done = False

        while not done:
            if env.current_player == as_player:
                action = agent.select_action(env.board, env.current_player)  # MCTS move
            else:
                action = policy_opponent(env)  # opponent policy
            _, _, done = env.step(action)

        winner = env.winner()
        if winner == as_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    win_rate = wins / n_games
    draw_rate = draws / n_games
    loss_rate = losses / n_games

    return win_rate, draw_rate, loss_rate

def evaluate_vs_random(
    env: TicTacToeEnv,
    agent: MCTSAgent,
    n_games: int = 200,
    as_player: int = 1,
) -> tuple[float, float, float]:
    """Evaluate MCTS win/draw/loss rates against a uniform-random opponent.

    Args:
        env:       TicTacToe environment instance.
        agent:     MCTSAgent to evaluate.
        n_games:   number of evaluation games.
        as_player: +1 → MCTS plays X (first mover); -1 → MCTS plays O.

    Returns:
        ``(win_rate, draw_rate, loss_rate)`` — three fractions summing to 1.
    """
    return evaluate_vs_policy(
        env,
        agent,
        random_action,
        n_games=n_games,
        as_player=as_player
    )
