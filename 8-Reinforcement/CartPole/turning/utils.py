import random

import gymnasium as gym
import numpy as np


_actions = (0, 1)


def discretize(x):
    return tuple((x / np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int32))


def create_bins(i, num):
    return np.linspace(i[0], i[1], num + 1)


ints = [(-5, 5), (-2, 2), (-0.5, 0.5), (-2, 2)]
nbins = [20, 20, 10, 10]
bins = [create_bins(ints[i], nbins[i]) for i in range(4)]


def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


#: name -> callable. Names are what gets stored inside a policy bundle.
DISCRETIZERS = {"fine": discretize, "coarse": discretize_bins}


def resolve_discretizer(name):
    """Accept a name ('fine'/'coarse') or a callable and return the callable."""
    if callable(name):
        return name
    try:
        return DISCRETIZERS[name]
    except KeyError:
        raise ValueError(
            f"unknown discretizer {name!r}; expected one of {sorted(DISCRETIZERS)} "
            "or a callable"
        ) from None


def discretizer_name(drz):
    """Inverse of :func:`resolve_discretizer`, for saving."""
    if isinstance(drz, str):
        return drz
    for name, fn in DISCRETIZERS.items():
        if fn is drz:
            return name
    return getattr(drz, "__name__", "custom")


def _probs(v, eps=1e-4):
    """
    Convert a weight vector into probs
    """
    v = v - v.min() + eps
    v = v / v.sum()
    return v


def fetch_action_based_Q(Q, space, discretizer=None):
    """Return a function mapping an observation to an action via the Q-table.

    ``discretizer`` MUST match the grid the table was trained on. Passing the
    wrong one is silent: every lookup misses, both Q-values read as 0, and the
    returned policy degenerates to uniform random actions.
    """
    drz = resolve_discretizer(discretizer or "fine")

    def fetch(state):
        s = drz(state)
        candidates = [Q.get((s, a), 0) for a in space]
        v = _probs(np.array(candidates))
        return random.choices(space, weights=v)[0]

    return fetch


def play_with_Q(policy, env: gym.Env, episodes=1, verbose=True):
    """
    You should set the render_mode before call this function.
    Returns the list of episode returns.
    """
    returns = []
    for _ in range(episodes):
        done = False
        observation, info = env.reset()
        cum_reward = 0
        while not done:
            na = policy(observation)
            observation, reward, terminated, truncated, info = env.step(na)
            cum_reward += reward
            done = terminated or truncated
        returns.append(cum_reward)
        if verbose:
            print(f"evaluating over, reward is: {cum_reward}")
    return returns
