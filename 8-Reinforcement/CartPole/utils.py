import random

import gymnasium as gym
import numpy as np


actions = (0, 1)


def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int32))


def create_bins(i, num):
    return np.linspace(i[0], i[1], num + 1)


ints = [(-5, 5), (-2, 2), (-0.5, 0.5), (-2, 2)]
nbins = [20, 20, 10, 10]
bins = [create_bins(ints[i], nbins[i]) for i in range(4)]


def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


def _probs(v, eps=1e-4):
    """
    Convert a weight vector into probs
    """
    v = v - v.min() + eps
    v = v / v.sum()
    return v


def fetch_action_based_Q(Q, space):
    """
    return a function receive current observation then return choice based on Q
    """

    def fetch(state):
        s = discretize(state)
        candidates = [Q.get((s, a), 0) for a in space]
        v = _probs(np.array(candidates))
        return random.choices(space, weights=v)[0]

    return fetch


def play_with_Q(policy, env: gym.Env):
    """
    You should set the render_mode before call this function
    """

    done = False

    observation, info = env.reset()

    cum_reward = 0
    while not done:
        na = policy(observation)

        observation, reward, terminated, truncated, info = env.step(na)
        cum_reward += reward

        done = terminated or truncated

    print(f"evaluating over, reward is: {cum_reward}")
