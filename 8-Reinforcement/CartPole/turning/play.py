"""Load a saved policy and play it.

Handles both the self-describing ``*.policy.pkl`` bundles and the legacy bare
``*.plk`` Q-tables (which carry no grid metadata -- use ``--discretizer`` to
tell it which one the table was trained with).

    python play.py runs/xxx.policy.pkl --episodes 20
    python play.py runs/xxx.policy.pkl --render
    python play.py runs/old.plk --discretizer coarse --episodes 20
    python play.py runs/old.plk --sweep          # diagnose a grid mismatch
"""

import argparse
import os

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "mpl")
)

import numpy as np

from model import load_policy
from utils import DISCRETIZERS, fetch_action_based_Q, resolve_discretizer


def rollout(policy, episodes, render=False, seed0=20_000_000):
    import gymnasium as gym

    try:
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
    except Exception as exc:
        print(f"render unavailable ({exc}); running headless")
        env = gym.make("CartPole-v1")

    out = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed0 + i)
            done, total = False, 0.0
            while not done:
                obs, r, te, tr, _ = env.step(policy(obs))
                total += r
                done = te or tr
            out.append(total)
    finally:
        env.close()
    return np.asarray(out, dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="policy bundle or legacy Q-table pickle")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--discretizer", choices=sorted(DISCRETIZERS),
                   help="override the grid (legacy tables, or to reproduce a mismatch)")
    p.add_argument("--seed0", type=int, default=20_000_000)
    p.add_argument("--sweep", action="store_true",
                   help="try every discretizer and report which one this table actually fits")
    args = p.parse_args()

    Q, cfg, drz_name = load_policy(args.path)
    print(f"policy   {args.path}")
    print(f"entries  {len(Q)}")
    print(f"grid     {drz_name}  (from bundle metadata)")
    if cfg is not None:
        print(f"config   {cfg}")
    else:
        print("config   unknown (legacy bare Q-table)")

    names = sorted(DISCRETIZERS) if args.sweep else [args.discretizer or drz_name]

    for name in names:
        drz = resolve_discretizer(name)
        pol = fetch_action_based_Q(Q, (0, 1), discretizer=drz)
        ret = rollout(pol, args.episodes, render=args.render and len(names) == 1,
                      seed0=args.seed0)
        flag = ""
        if cfg is not None and name != drz_name:
            flag = "   <-- grid mismatch (not the grid this table was trained on)"
        elif cfg is None and name == drz_name:
            flag = "   <-- grid you asked for (legacy table: unverified)"
        print(f"\ngrid {name:6s}  mean {ret.mean():7.2f}  "
              f"median {np.median(ret):6.1f}  min {ret.min():5.0f}  max {ret.max():5.0f}{flag}")

    if args.sweep:
        print("\nA table whose own grid scores near the random-policy baseline (~20) "
              "means the lookup is missing; the higher row is the one it was trained on.")


if __name__ == "__main__":
    main()
