import os
import csv
from collections import deque
import matplotlib.pyplot as plt
import numpy as np



def rolling_mean(values, window: int):
    """Compute rolling mean with a fixed window size."""
    out = []
    q = deque(maxlen=window)
    s = 0.0
    for v in values:
        if len(q) == q.maxlen:
            s -= q[0]
        q.append(v)
        s += v
        out.append(s / len(q))
    return out


def plot_learning_curve(
    csv_path: str = "results/tabular_log.csv",
    out_path: str = "resultsr/tabular_learning_curve.png",
    window: int = 100,
    show: bool = False,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    episodes, returns, lengths, epsilons = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
            lengths.append(int(row["length"]))
            epsilons.append(float(row["epsilon"]))

    returns_rm = rolling_mean(returns, window=window)
    lengths_rm = rolling_mean(lengths, window=window)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Figure 1: Return curve
    plt.figure()
    plt.plot(episodes, returns, label="Episode return")
    plt.plot(episodes, returns_rm, label=f"Rolling mean (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Tabular Q-learning: Learning Curve episodes=5000")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    # Figure 2: Episode length curve (optional but useful)
    out_path_len = out_path.replace(".png", "_length.png")
    plt.figure()
    plt.plot(episodes, lengths, label="Episode length")
    plt.plot(episodes, lengths_rm, label=f"Rolling mean (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Tabular Q-learning: Learning Curve (Episode Length)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_len, dpi=200)
    if show:
        plt.show()
    plt.close()

    print(f"Saved: {out_path}")
    print(f"Saved: {out_path_len}")


if __name__ == "__main__":
    plot_learning_curve()          #  tabular 