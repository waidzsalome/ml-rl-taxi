import numpy as np


class TabularQAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1, #learning rate
        gamma: float = 0.99,
        epsilon: float = 1.0, # ε range from 0 to 1
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,# factor to decrease epsilon
        seed: int | None = None,
    ):
        self.n_states = n_states # 500
        self.n_actions = n_actions # 6
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        ## initial Q table, 500x6, intital value: 0
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    # epsilon-greedy  
    def act(self, state: int) -> int:
        # exploration
        # select a random action with probability ε 
        # self.rng.random() return a RN range from 0 to 1
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # exploitation, select action a that maximizes Q(s, a)
        # current state, value of Q_table
        q = self.Q[state]
        # max value of Q(s,?)
        max_q = np.max(q)
        # list all of maxQ(s,?),sometime maybe equal
        best_actions = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best_actions))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        if done:
            # no future rewards
            target = r
        else:
            target = r + self.gamma * float(np.max(self.Q[s2]))

        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay_eps(self):
        # Multiply ε by a factor smaller than 1 to gradually decrease the exploration
        # but no smaller than min, which is bigger than 0
        # (no exploration, once Q table has errors,unable to correct )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    # save Q table
    def save(self, path: str):
        np.save(path, self.Q)
    # load Q table
    def load(self, path: str):
        self.Q = np.load(path)
