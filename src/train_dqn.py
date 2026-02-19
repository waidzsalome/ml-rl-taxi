from __future__ import annotations

import os
from dataclasses import asdict
import numpy as np
import torch

from src.utils.env import make_taxi_env
from src.agents.dqn import DQNAgent, DQNConfig

def main():
    # ---- hyperparams ----
    seed = 42
    episodes = 2000
    max_steps_per_ep = 200

    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer=2_000,
        train_every=1,
        grad_clip=10.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
    )

    # ---- env ----
    env = make_taxi_env(seed=seed)

    assert hasattr(env.observation_space, "n"), "Taxi-v3 should have discrete observation_space.n"
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # ---- agent ----
    agent = DQNAgent(n_states=n_states, n_actions=n_actions, cfg=cfg)

    # ---- logging ----
    returns = []
    losses = []

    # results dir (match your project layout)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # save config for report/repro
    with open(os.path.join(results_dir, "dqn_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\n")
        for k, v in asdict(cfg).items():
            f.write(f"{k}={v}\n")

    # ---- training loop ----
    for ep in range(1, episodes + 1):
        # s = int(reset_env(env))
        s,_ = env.reset()
        ep_ret = 0.0
        ep_losses = []
        done= False
        steps=0

        # for _ in range(max_steps_per_ep):
        while not done and steps< max_steps_per_ep:
            a = agent.act(s)  # epsilon-greedy inside
            s2, r, terminated, truncated, _ =env.step(a);
            done = terminated or truncated

            agent.remember(s, a, float(r), int(s2), done)

            loss = agent.learn_step()
            if loss is not None:
                ep_losses.append(loss)

            agent.step()  # increments total_steps for epsilon schedule

            ep_ret += float(r)
            s = int(s2)

            steps+=1

        returns.append(ep_ret)
        if len(ep_losses) > 0:
            losses.append(float(np.mean(ep_losses)))

        # print progress
        if ep % 50 == 0:
            avg_ret = float(np.mean(returns[-50:]))
            eps = agent.epsilon()
            avg_loss = float(np.mean(losses[-50:])) if len(losses) >= 50 else (float(np.mean(losses)) if losses else float("nan"))
            print(f"EP {ep:4d} | avg_return(last50)={avg_ret:7.2f} | eps={eps:.3f} | avg_loss={avg_loss:.4f}")

    env.close()

    # ---- save curves ----
    np.save(os.path.join(results_dir, "dqn_returns.npy"), np.array(returns, dtype=np.float32))
    np.save(os.path.join(results_dir, "dqn_losses.npy"), np.array(losses, dtype=np.float32))

    # ---- save model ----
    torch.save(agent.qnet.state_dict(), os.path.join(results_dir, "dqn_single_nn.pt"))

    print("Training done.")
    print(f"Saved: {results_dir}/dqn_returns.npy, dqn_losses.npy, dqn_single_nn.pt")


if __name__ == "__main__":
    main()