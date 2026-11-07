"""Tabular Q-learning on CartPole-v1 over a discretized state space.

The training entry point is deliberately re-entrant: every call owns its own
Q-table, its own RNG and its own reward log, so a hyperparameter search can run
thousands of trials in one process without contaminating them.
"""

import argparse
import json
import os
import pickle
import random
import time

import gymnasium as gym
import numpy as np

from model import Config, Result, save_policy
from utils import discretizer_name, resolve_discretizer

#: Evaluation seeds live far above any training seed range, so an eval episode
#: can never reuse an initial state the policy was trained on.
EVAL_SEED_OFFSET = 10_000_000

actions = (0, 1)

# Module-level table, kept only for backward compatibility with
# utils.play_with_Q / utils.fetch_action_based_Q. train() does NOT use it.
Q = {}


def make_env(render_mode=None):
    return gym.make("CartPole-v1", render_mode=render_mode)


def qvalues(state, table=None):
    t = Q if table is None else table
    return [t.get((state, a), 0.0) for a in actions]


def probs(v, eps=1e-4):
    """Convert a value vector into a probability vector."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    v = v - v.min() + eps
    total = v.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(v, 1.0 / v.size)
    return v / total


def greedy_action(v, rng):
    """Argmax with random tie-breaking (a zero table then acts uniformly)."""
    best = max(v)
    cands = [a for a, val in zip(actions, v) if val == best]
    return cands[0] if len(cands) == 1 else rng.choice(cands)


def train(
    env,
    c: Config,
    total_epochs=300000,
    batch=10,
    seed=None,
    discretizer=None,
    log_every=0,
):
    """Run epsilon-greedy Q-learning and return a :class:`model.Result`.

    The best Q-table is the one whose ``batch``-window mean was highest, stored
    as a *copy* (the old code aliased the live table, so "best" silently kept
    changing after the fact).
    """
    drz = resolve_discretizer(discretizer or "fine")
    rng = random.Random(seed)
    table = {}
    rewards = []
    window = []
    best_avg = float("-inf")
    best_Q = {}
    best_at = None
    t0 = time.time()

    for epoch in range(total_epochs):
        obs, _ = env.reset(**({} if seed is None else {"seed": seed + epoch}))
        alpha = c.alpha_func(epoch)
        episode_reward = 0.0
        done = False
        while not done:
            s = drz(obs)
            if rng.random() < c.epsilon:
                a = rng.choice(actions)
            else:
                a = random.choices(actions, weights=probs(qvalues(s, table)))[0]

            obs, reward, terminated, truncated, _ = env.step(a)
            episode_reward += reward
            done = terminated or truncated
            ns = drz(obs)

            if done:
                target = c.punish
            else:
                target = reward + c.gamma * max(qvalues(ns, table))

            cur = table.get((s, a), 0.0)
            table[(s, a)] = (1.0 - alpha) * cur + alpha * target

        rewards.append(episode_reward)
        window.append(episode_reward)

        if (epoch + 1) % batch == 0:
            avg = float(np.mean(window))
            if avg > best_avg:
                best_avg = avg
                best_Q = dict(table)
                best_at = epoch + 1
            if log_every and (epoch + 1) % log_every == 0:
                print(
                    f"{epoch + 1:>7d}: last-{batch} mean {avg:8.2f}  table {len(table)}"
                )
            window = []

    if not best_Q:
        best_Q = dict(table)
        best_avg = float(np.mean(rewards[-batch:])) if rewards else float("nan")

    return Result(
        c,
        best_Q,
        rewards,
        best_avg=best_avg,
        at_epoch=best_at,
        seed=seed,
        total_epochs=total_epochs,
        elapsed=time.time() - t0,
        discretizer=discretizer_name(drz),
    )


def evaluate(
    env, table, episodes=30, seed=EVAL_SEED_OFFSET, epsilon=0.0, discretizer=None
):
    """Roll out a learned Q-table; epsilon=0 means greedy (best learned policy)."""
    drz = resolve_discretizer(discretizer or "fine")
    rng = random.Random(seed)
    out = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        while not done:
            s = drz(obs)
            if rng.random() < epsilon:
                a = rng.choice(actions)
            else:
                a = greedy_action(qvalues(s, table), rng)
            obs, reward, terminated, truncated, _ = env.step(a)
            total += reward
            done = terminated or truncated
        out.append(total)
    return np.asarray(out, dtype=float)


def main():
    p = argparse.ArgumentParser(description="Train one Q-learning configuration.")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument(
        "--strategy",
        default="constant",
        choices=["constant", "linear", "exp", "invsqrt", "step"],
    )
    p.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.5],
        help="schedule parameters in declaration order, e.g. --strategy linear --alpha 0.5 0.02 4000",
    )
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.09)
    p.add_argument("--punish", type=float, default=-10.0)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--discretizer", default="fine", choices=["fine", "coarse"])
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    from model import ALPHA_STRATEGIES

    drz = resolve_discretizer(args.discretizer)

    names = ALPHA_STRATEGIES[args.strategy][1]
    if len(args.alpha) > len(names):
        p.error(
            f"--strategy {args.strategy} takes at most {len(names)} params: {names}"
        )
    params = dict(zip(names, args.alpha))

    c = Config.build(
        name="single",
        strategy=args.strategy,
        alpha_params=params,
        gamma=args.gamma,
        epsilon=args.epsilon,
        punish=args.punish,
    )
    env = make_env()
    try:
        r = train(
            env,
            c,
            total_epochs=args.episodes,
            batch=args.batch,
            seed=args.seed,
            discretizer=drz,
            log_every=500,
        )
        ev = evaluate(
            env,
            r.Qbest,
            episodes=args.eval_episodes,
            seed=EVAL_SEED_OFFSET,
            discretizer=drz,
        )
    finally:
        env.close()

    print(f"config          {c}")
    print(f"discretizer     {args.discretizer}")
    print(f"train window    {r.final_window(100):.2f}")
    print(f"best batch mean {r.best_avg:.2f} @ {r.at_epoch}")
    print(
        f"greedy eval     mean {ev.mean():.2f}  median {np.median(ev):.1f}  min {ev.min():.0f}  max {ev.max():.0f}"
    )
    print(f"elapsed         {r.elapsed:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    bundle = r.save_policy(args.out)
    r.window_plot(args.out, window=200)
    with open(f"{args.out}/{r.config.slug()}.json", "w") as f:
        json.dump({**r.to_dict(), "greedy_eval_mean": float(ev.mean())}, f, indent=2)
    print(f"\nsaved           {bundle}")
    print(f"play it with    python play.py {bundle} --episodes 20")


if __name__ == "__main__":
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "mpl"),
    )
    main()
