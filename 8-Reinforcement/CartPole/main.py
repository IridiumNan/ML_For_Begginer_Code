import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import matplotlib.pyplot as plt
import numpy as np
import random


env = gym.make("CartPole-v1", render_mode="human")


observation, info = env.reset()

print(f"Starting observation: {observation}")

# the shape of observation_space is (4,)
# it contains the following values:
# - Position of cart
# - Velocity of cart
# - Angle of cart
# - Rotation rate of pole

episode_over = False
total_reward = 0


# NOTE: This is a demo
# single loop
# while True:
#     # randomly select an action
#     action = env.action_space.sample()
#
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     # reward +1 for each step the pole stay upright
#     # terminated: True if pole falls too far (agent failed)
#     # truncated: True if we hit the time limit (500 steps)
#     total_reward += reward
#     episode_over = terminated or truncated
#
#     if episode_over:
#         break


Q = {}


def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]


# hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.09


def probs(v, eps=1e-4):
    """
    Convert a weight vector into probs
    """
    v = v - v.min() + eps
    v = v / v.sum()
    return v


Qmax = 0
cum_rewards = []
rewards = []

# no display when training
# This is invalid
# env.render_mode = None
# create a new one instead
env.close()

env = gym.make(
    "CartPole-v1",
)

from utils import *

# play_with_Q(Q, env)

for epoch in range(30000):
    observation, info = env.reset()
    done = False
    cum_reward = 0

    # do simulation
    while not done:
        # s is the current state
        s = discretize(observation)

        if random.random() < epsilon:
            # random explore
            a = random.choices(actions)[0]
            # a = np.random.randint(env.action_space.n)
        else:
            # exploitation - chose the actoin according to Q-Table proba
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions, weights=v)[0]

        observation, reward, terminated, truncated, info = env.step(a)
        cum_reward += reward
        # ns is the next state
        ns = discretize(observation)
        done = terminated or truncated
        target = reward + gamma * max(qvalues(ns)) if not done else -10
        # Q[(s, a)] = (1 - alpha) * Q.get((s, a), 0) + alpha * target
        Q[(s, a)] = (1 - alpha) * Q.get((s, a), 0) + alpha * target

    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    if epoch % 300 == 0:
        print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards = []

# close train env then create a new one


env.close()

import pickle

with open("Q.pkl", "wb") as f:
    pickle.dump(Qbest, f)
