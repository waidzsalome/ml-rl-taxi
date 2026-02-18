# eval_dqn.py
import os
import csv
import numpy as np
import torch

from src.utils.env import make_taxi_env
from src.agents.dqn import DQNAgent, DQNConfig


def evaluate(
    seed: int = 42,
    num_episodes: int = 100,
    max_steps: int = 200,
    model_path: str = "results/dqn_single_nn.pt",
    results_dir: str = "results",
    render: bool = False,
) -> dict:
    """
    Evaluate a trained DQN agent (greedy policy) on Taxi-v3.

    What it does:
    - load trained Q-network
    - run num_episodes episodes with greedy actions (epsilon = 0)
    - report average return, average length, success rate
    """

    os.makedirs(results_dir, exist_ok=True)

    env = make_taxi_env(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = DQNAgent(
        n_states=n_states,
        n_actions=n_actions,
        cfg=DQNConfig(),
    )

    # load trained network
    agent.qnet.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.qnet.eval()

    returns = []
    lengths = []
    successes = 0  # Taxi: success = terminated (not truncated)

    for ep in range(1, num_episodes + 1):
        s, _ = env.reset()
        s = int(s)

        done = False
        success_done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            # greedy action
            a = agent.act(s, greedy=True)

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            success_done = terminated

            ep_return += float(r)
            steps += 1
            s = int(s2)

            if render:
                try:
                    env.render()
                except Exception:
                    pass

        returns.append(ep_return)
        lengths.append(steps)

        if success_done:
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
        "model_path": model_path,
        "seed": seed,
    }

    # ---- save summary txt ----
    summary_txt_path = os.path.join(results_dir, "dqn_eval_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== DQN Evaluation Summary ===\n")
        f.write(f"Episodes:      {num_episodes}\n")
        f.write(f"Avg return:    {avg_return:.3f}\n")
        f.write(f"Avg length:    {avg_len:.3f}\n")
        f.write(f"Success rate:  {success_rate:.2%}\n")
        f.write(f"Model path:    {model_path}\n")
        f.write(f"Seed:          {seed}\n")

    # ---- save per-episode csv ----
    eval_log_path = os.path.join(results_dir, "dqn_eval.csv")
    with open(eval_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length"])
        for i, (ret, ln) in enumerate(zip(returns, lengths), start=1):
            writer.writerow([i, ret, ln])

    print("\n=== DQN Evaluation Summary ===")
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
