import numpy as np
from src.utils.env import make_taxi_env

def eval_tabular(q_path="results/tabular_q.npy", seed=0, episodes=100, max_steps=200):
    env = make_taxi_env(seed=seed)
    Q = np.load(q_path)

    returns = []
    lengths = []

    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            a = int(np.argmax(Q[int(s)]))  # greedy
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_return += float(r)
            steps += 1

        returns.append(ep_return)
        lengths.append(steps)

    env.close()
    print(f"Eval episodes: {episodes}")
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")

if __name__ == "__main__":
    eval_tabular()
