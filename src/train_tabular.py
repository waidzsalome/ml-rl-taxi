import os
import csv
from src.utils.env import make_taxi_env
from src.agents.tabular_q import TabularQAgent


def train(
    seed: int = 42,
    num_episodes: int = 20000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    results_dir: str = "results",
):
    os.makedirs(results_dir, exist_ok=True)

    env = make_taxi_env(seed=seed)
    nS = env.observation_space.n
    nA = env.action_space.n

    agent = TabularQAgent(
        n_states=nS,
        n_actions=nA,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    log_path = os.path.join(results_dir, "tabular_log.csv")
    q_path = os.path.join(results_dir, "tabular_q.npy")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length", "epsilon"])

        for ep in range(1, num_episodes + 1):
            s, _ = env.reset()
            done = False
            ep_return = 0.0
            steps = 0

            while not done and steps < max_steps:
                a = agent.act(int(s))
                s2, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                agent.update(int(s), int(a), float(r), int(s2), done)

                s = s2
                ep_return += float(r)
                steps += 1

            agent.decay_eps()
            writer.writerow([ep, ep_return, steps, agent.epsilon])

            # 每 500 回合打印一下，方便你看到在学
            if ep % 500 == 0:
                print(f"[Tabular] ep={ep} return={ep_return:.1f} len={steps} eps={agent.epsilon:.3f}")

    agent.save(q_path)
    env.close()
    print(f"Saved log -> {log_path}")
    print(f"Saved Q-table -> {q_path}")


def main():
    train()


if __name__ == "__main__":
    main()
