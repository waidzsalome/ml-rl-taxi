# ml-rl-taxi

## introduction

## problem addressed

### env des

The problem addressed in this project consists in training a reinforcement learning agent to solve the Taxi-v3 environment provided by the Gymnasium framework.

### RL

deterministic
MDP

action space 6
state space 500
reward

## the solution adopted

### Tabular Q-Learning

#### update

$$
Q(s, a) = r(s, a) + \gamma \max_{a' \in A} Q(\delta(s, a), a')
$$

add temporal difference, with lamda=0,and learning rate

$$
\hat{Q}(s, a) \leftarrow (1-\alpha)\hat{Q}(s, a) + \alpha [r + \gamma \max_{a'} \hat{Q}(s', a')]
$$

$$
\hat{Q}(s,a) \leftarrow \hat{Q}(s,a)+ \underbrace{\alpha}_{\text{learning rate}} \Big[\underbrace{r + \overbrace{\gamma}^{\text{discount factor}}\max_{a'} \hat{Q}(s',a')}_{\text{TD target}}- \hat{Q}(s,a)\Big]
$$

#### choosing action, greedy

#### algorithm

### DQN

#### update

#### choosing action

#### algorithm

## the experimental results

### Tabular Q-Learning

#### Learning curve

#### different eposides

#### different lr

#### different gamma

### DQN

### learning curve

### comparation Tabular-Q Learning and DQN

## the architecture of the implementation.

### Tabular

### DQN
