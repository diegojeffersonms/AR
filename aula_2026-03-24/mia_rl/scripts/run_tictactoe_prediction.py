from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tic-Tac-Toe model-free prediction experiments.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0).")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/tictactoe_prediction", help="Directory inside mia_rl where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from mia_rl.agents.prediction import FirstVisitMonteCarloPrediction, TD0Prediction
    from mia_rl.experiments.training import generate_episode, train_prediction_agent
    from mia_rl.envs.tictactoe import TicTacToeEnv

    try:
        sample_env = TicTacToeEnv()
        sample_episode = generate_episode(sample_env, policy=None)  # TODO: add a random policy
        print(f"Sample episode length: {len(sample_episode.transitions)}")
        print("First transitions:")
        for transition in sample_episode.transitions[:5]:
            print(transition)

        mc_env = TicTacToeEnv()
        td_env = TicTacToeEnv()
        mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)

        checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

        print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
        mc_history = train_prediction_agent(mc_env, policy=None, agent=mc_agent, num_episodes=args.episodes, checkpoints=checkpoints)

        print(f"Training TD(0) for {args.episodes} episodes...")
        td_history = train_prediction_agent(td_env, policy=None, agent=td_agent, num_episodes=args.episodes, checkpoints=checkpoints)
    except Exception as e:
        print(f"Error during training: {e}")
        return
