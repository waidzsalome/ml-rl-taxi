import os
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

def plot_dqn_curve(
    returns_path: str = "results/dqn_returns.npy",
    losses_path: str = "results/dqn_losses.npy",
    out_prefix: str = "results/dqn_learning_curve",
    window: int = 100,
    show: bool = False,
):
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"NPY not found: {returns_path}")

    returns = np.load(returns_path).tolist()
    episodes = list(range(1, len(returns) + 1))
    returns_rm = rolling_mean(returns, window=window)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # Figure 1: Return curve
    out_path = f"{out_prefix}.png"
    plt.figure()
    plt.plot(episodes, returns, label="Episode return")
    plt.plot(episodes, returns_rm, label=f"Rolling mean (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN (single NN): Learning Curve Ep=5000 alpha=0.1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {out_path}")

    # Figure 2: Loss curve (optional)
    if os.path.exists(losses_path):
        losses = np.load(losses_path).tolist()
        if len(losses) > 0:
            loss_eps = list(range(1, len(losses) + 1))
            losses_rm = rolling_mean(losses, window=min(window, len(losses)))

            out_path_loss = f"{out_prefix}_loss.png"
            plt.figure()
            plt.plot(loss_eps, losses, label="Loss")
            plt.plot(loss_eps, losses_rm, label=f"Rolling mean (window={min(window, len(losses))})")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("DQN (single NN): Training Loss Ep=5000 alpha=0.1")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path_loss, dpi=200)
            if show:
                plt.show()
            plt.close()
            print(f"Saved: {out_path_loss}")


if __name__ == "__main__":       
    plot_dqn_curve()        # dqn