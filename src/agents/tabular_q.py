import numpy as np


class TabularQAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def act(self, state: int) -> int:
        # epsilon-greedy
        # exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # exploitation
        q = self.Q[state]
        max_q = np.max(q)
        # 随机打破并列最大值（避免总选第一个动作）
        best_actions = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.Q[s2]))

        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)
