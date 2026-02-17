# eval_tabular.py
import os
import csv
from typing import Optional

import numpy as np

from src.utils.env import make_taxi_env
from src.agents.tabular_q import TabularQAgent


def evaluate(
    seed: int = 42,
    num_episodes: int = 100,
    max_steps: int = 200,
    q_path: str = "results/tabular_q.npy",
    results_dir: str = "results",
    render: bool = False,
) -> dict:
    """
    Evaluate a trained tabular Q-learning agent (greedy policy) on Taxi.

    What it does:
    - load Q-table from q_path
    - run num_episodes episodes with greedy actions (epsilon = 0)
    - report average return, average length, success rate

    """
    os.makedirs(results_dir, exist_ok=True)


    env = make_taxi_env(seed=seed)

    nS = env.observation_space.n
    nA = env.action_space.n

    agent = TabularQAgent(n_states=nS, n_actions=nA, seed=seed)
    agent.load(q_path)

    # Pure greedy evaluation (no exploration)
    agent.epsilon = 0.0

    returns = []
    lengths = []
    successes = 0  # Taxi: success typically means episode ends with terminitaion, no truncation

    for ep in range(1, num_episodes + 1):
        s, _ = env.reset()
        done = False
        successe_done=False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            # Greedy action from Q-table (with random tie-breaking inside act)
            a = agent.act(int(s))

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            successe_done = terminated

            ep_return += float(r)
            steps += 1
            s = s2

            if render:
                # Many Gymnasium envs need render_mode set at creation time.
                # If your env supports env.render() here, it will show.
                try:
                    env.render()
                except Exception:
                    pass

        returns.append(ep_return)
        lengths.append(steps)

        # Heuristic for Taxi-v3: successful dropoff gives +20 and ends episode.
        # If your reward scheme differs, adapt this.
        if successe_done:
            successes += 1

        if ep % max(1, num_episodes // 10) == 0:
            print(f"[Eval] ep={ep}/{num_episodes} return={ep_return:.1f} len={steps}")

    env.close()

    avg_return = float(np.mean(returns)) if returns else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    success_rate = successes / num_episodes if num_episodes > 0 else 0.0

    summary = {
        "num_episodes": num_episodes,
        "avg_return": avg_return,
        "avg_length": avg_len,
        "success_rate": success_rate,
        "q_path": q_path,
        "seed": seed,
    }

    # Save per-episode eval log (optional but useful)
    eval_log_path = os.path.join(results_dir, "tabular_eval.csv")
    with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        for i, (ret, ln) in enumerate(zip(returns, lengths), start=1):
            writer.writerow([i, ret, ln])

    print("\n=== Evaluation Summary ===")
    print(f"Episodes:      {num_episodes}")
    print(f"Avg return:    {avg_return:.3f}")
    print(f"Avg length:    {avg_len:.3f}")
    print(f"Success rate:  {success_rate:.2%}")
    print(f"Saved eval log -> {eval_log_path}")

    return summary


def main():
    evaluate()


if __name__ == "__main__":
    main()
