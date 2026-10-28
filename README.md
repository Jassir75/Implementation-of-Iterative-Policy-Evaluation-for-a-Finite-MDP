# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python

import gymnasium as gym
import numpy as np

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True
)

env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

policy = np.ones((n_states, n_actions)) / n_actions

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    
    # Initialize value function
    V = np.zeros(env.observation_space.n)

    iteration = 0

    while True:
        delta = 0

        # Create a copy so that all states in this iteration
        # use the previous value function
        new_V = np.zeros_like(V)

        # Evaluate every state
        for s in range(env.observation_space.n):

            v = 0

            # Consider every possible action
            for a in range(env.action_space.n):

                # Probability of selecting action a
                action_prob = policy[s, a]

                # Environment transitions
                for prob, next_state, reward, terminated in env.P[s][a]:

                    # If the next state is terminal,
                    # there is no future value
                    future_value = 0 if terminated else V[next_state]

                    v += action_prob * prob * (
                        reward + gamma * future_value
                    )

            new_V[s] = v

            # Track maximum change
            delta = max(delta, abs(V[s] - new_V[s]))

        # Update value function
        V = new_V

        iteration += 1

        # Check convergence
        if delta < theta:
            break

    return V, iteration


V, iterations = policy_evaluation(
    env,
    policy,
    gamma,
    theta
)

print("Name: Jassir Sulthan")
print("Register Number: 212224240060")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name: Jassir Sulthan")
print("Register Number: 212224240060")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()

```

---

## Output

<img width="1024" height="591" alt="image" src="https://github.com/user-attachments/assets/54ff05e1-a33b-45a9-abc0-44a2c2b3749d" />


---
## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

1.The value function converged in 46 iterations, showing that iterative policy evaluation successfully reached convergence under the fixed random policy.

2.Hole states (5, 7, 11, and 12) and the goal state (15) have V = 0.0, as they are terminal states with no future rewards after the episode ends.

3.State 14 has the highest value (0.4130) because it is adjacent to the goal and has the highest probability of receiving the reward.

4.The state values gradually increase as the agent moves closer to the goal, forming a clear value gradient across the 4×4 FrozenLake grid.

5.States near the starting position have very low values, indicating that a random policy has a low probability of reaching the goal without falling into a hole.

6.Since is_slippery=True, the environment is stochastic, meaning the agent may move in unintended directions. This uncertainty lowers the expected value of most states compared to a deterministic environment.

---
