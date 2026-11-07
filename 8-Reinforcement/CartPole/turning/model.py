"""Hyperparameter containers and learning-rate (alpha) schedules.

Every schedule is a factory returning ``alpha_func(n) -> float`` where ``n`` is
the episode index.  Schedules are clamped to a numerically safe range so a
wide hyperparameter search can never blow the Q-table up.
"""

import os

import numpy as np


# --------------------------------------------------------------- schedules
_ALPHA_MIN = 1e-6
_ALPHA_MAX = 1.0


def _clamp(a):
    return float(min(_ALPHA_MAX, max(_ALPHA_MIN, a)))


def const_alpha(alpha):
    """Constant learning rate."""
    a = float(alpha)

    def alpha_func(n):
        return _clamp(a)

    return alpha_func


def default_alpha(n):
    """Backward-compatible default: constant 0.5.

    NOTE: the previous version returned ``const_alpha(0.5)`` -- a *function* --
    which made ``(1 - alpha)`` raise TypeError.
    """
    return 0.5


def linear_alpha(a0, a1=0.0, horizon=5000):
    """Linearly anneal a0 -> a1 over ``horizon`` episodes."""
    a0, a1 = float(a0), float(a1)
    horizon = max(1, int(horizon))

    def alpha_func(n):
        frac = min(1.0, max(0.0, n / horizon))
        return _clamp(a0 + (a1 - a0) * frac)

    return alpha_func


def exp_alpha(a0, decay=0.999, floor=1e-3):
    """Geometric decay, ``a0 * decay**n``, never below ``floor``."""
    a0, decay, floor = float(a0), float(decay), float(floor)

    def alpha_func(n):
        return _clamp(max(floor, a0 * (decay**n)))

    return alpha_func


# 0.8283 367.9 0.001
def invsqrt_alpha(a0, scale=1000.0, floor=1e-3):
    """``a0 / sqrt(1 + n / scale)`` -- slow polynomial decay."""
    a0, floor = float(a0), float(floor)
    scale = max(1.0, float(scale))

    def alpha_func(n):
        return _clamp(max(floor, a0 / np.sqrt(1.0 + n / scale)))

    return alpha_func


def step_alpha(a0, factor=0.5, every=1000, floor=1e-3):
    """Multiply by ``factor`` every ``every`` episodes, never below ``floor``."""
    a0, factor, floor = float(a0), float(factor), float(floor)
    every = max(1, int(every))

    def alpha_func(n):
        return _clamp(max(floor, a0 * (factor ** (n // every))))

    return alpha_func


#: name -> (factory, tunable parameter names)
ALPHA_STRATEGIES = {
    "constant": (const_alpha, ("alpha",)),
    "linear": (linear_alpha, ("a0", "a1", "horizon")),
    "exp": (exp_alpha, ("a0", "decay", "floor")),
    "invsqrt": (invsqrt_alpha, ("a0", "scale", "floor")),
    "step": (step_alpha, ("a0", "factor", "every", "floor")),
}

DEFAULT_ALPHA_PARAMS = {
    "constant": {"alpha": 0.5},
    "linear": {"a0": 0.5, "a1": 0.0, "horizon": 5000},
    "exp": {"a0": 0.5, "decay": 0.999, "floor": 1e-3},
    "invsqrt": {"a0": 0.5, "scale": 1000.0, "floor": 1e-3},
    "step": {"a0": 0.5, "factor": 0.5, "every": 1000, "floor": 1e-3},
}


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        s = str(int(v))
    else:
        s = f"{float(v):.4g}"
    return s.replace("-", "m")


def make_slug(*parts):
    """Filesystem-safe identifier: no brackets, commas, equals or slashes.

    The old ``__str__`` style leaked ``[a0=0.83,scale=368]`` into file names,
    which breaks unquoted shell globbing.
    """
    s = "_".join(str(p) for p in parts if p not in (None, ""))
    for bad in "[](){}<>,=:/\\ \t\n'\"":
        s = s.replace(bad, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def build_alpha_func(strategy, params=None):
    """Instantiate a schedule, filling missing parameters with defaults."""
    if strategy not in ALPHA_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {sorted(ALPHA_STRATEGIES)}"
        )
    factory, names = ALPHA_STRATEGIES[strategy]
    merged = dict(DEFAULT_ALPHA_PARAMS[strategy])
    for k, v in (params or {}).items():
        if k not in names:
            raise ValueError(f"{strategy} has no parameter {k!r}; expected {names}")
        merged[k] = v
    # Integer-valued parameters are declared as such in DEFAULT_ALPHA_PARAMS.
    for k, default in DEFAULT_ALPHA_PARAMS[strategy].items():
        if isinstance(default, int) and merged[k] is not None:
            merged[k] = int(round(merged[k]))
    return factory(**merged)


class Config:
    def __init__(
        self,
        name: str,
        alpha_func=default_alpha,
        gamma=0.9,
        epsilon=0.09,
        punish=-10,
        strategy="constant",
        alpha_params=None,
    ):
        """
        name: label used in output file names
        alpha_func: func(n) -> learning rate for episode n
        gamma: discount factor
        epsilon: probability of a uniformly random action during training
        punish: terminal target (replaces reward + bootstrap at episode end)
        strategy: schedule name, kept for reporting and reconstruction
        alpha_params: dict of the schedule's parameters, kept for reporting
        """
        self.name = name
        self.alpha_func = alpha_func
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.punish = float(punish)
        self.strategy = strategy
        self.alpha_params = dict(alpha_params or {})

    @classmethod
    def build(
        cls,
        name="cfg",
        strategy="constant",
        alpha_params=None,
        gamma=0.9,
        epsilon=0.09,
        punish=-10.0,
    ):
        params = dict(DEFAULT_ALPHA_PARAMS[strategy], **(alpha_params or {}))
        return cls(
            name=name,
            alpha_func=build_alpha_func(strategy, params),
            gamma=gamma,
            epsilon=epsilon,
            punish=punish,
            strategy=strategy,
            alpha_params=params,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "strategy": self.strategy,
            "alpha_params": dict(self.alpha_params),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "punish": self.punish,
        }

    def slug(self):
        """Short, shell-safe name for files. Includes the discretizer-agnostic parts."""
        params = "_".join(f"{k}{_fmt(v)}" for k, v in sorted(self.alpha_params.items()))
        return make_slug(
            self.name,
            self.strategy,
            params,
            f"g{_fmt(self.gamma)}",
            f"e{_fmt(self.epsilon)}",
            f"p{_fmt(self.punish)}",
        )

    @classmethod
    def from_dict(cls, d):
        return cls.build(
            name=d.get("name", "cfg"),
            strategy=d["strategy"],
            alpha_params=d.get("alpha_params"),
            gamma=d.get("gamma", 0.9),
            epsilon=d.get("epsilon", 0.09),
            punish=d.get("punish", -10.0),
        )

    def __str__(self) -> str:
        params = ",".join(
            f"{k}={_fmt(v)}" for k, v in sorted(self.alpha_params.items())
        )
        return (
            f"{self.name}_{self.strategy}[{params}]"
            f"_g{_fmt(self.gamma)}_e{_fmt(self.epsilon)}_p{_fmt(self.punish)}"
        )


class Result:
    def __init__(
        self,
        c: Config,
        Qbest,
        rewards,
        best_avg=None,
        at_epoch=None,
        seed=None,
        total_epochs=None,
        elapsed=None,
        discretizer="fine",
    ) -> None:
        self.config = c
        self.Qbest = Qbest
        self.rewards = list(rewards)
        self.best_avg = best_avg
        self.at_epoch = at_epoch
        self.seed = seed
        self.total_epochs = total_epochs
        self.elapsed = elapsed
        self.discretizer = discretizer
        self.estimate = None

    def final_window(self, window=100):
        """Mean reward over the last ``window`` training episodes."""
        if not self.rewards:
            return float("nan")
        tail = self.rewards[-min(window, len(self.rewards)) :]
        return float(np.mean(tail))

    def to_dict(self):
        return {
            "config": self.config.to_dict(),
            "discretizer": self.discretizer,
            "best_avg": self.best_avg,
            "at_epoch": self.at_epoch,
            "seed": self.seed,
            "total_epochs": self.total_epochs,
            "elapsed": self.elapsed,
            "final_window": self.final_window(),
        }

    def window_plot(self, dir, window=200):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rewards = np.asarray(self.rewards, dtype=float)
        if rewards.size < window:
            raise ValueError(f"need at least {window} episodes, got {rewards.size}")

        windows = np.lib.stride_tricks.sliding_window_view(rewards, window)
        mean_vals = windows.mean(axis=1)
        max_vals = windows.max(axis=1)
        min_vals = windows.min(axis=1)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot()
        ax.plot(mean_vals, c="blue", label="Mean", linewidth=2)
        ax.plot(max_vals, c="green", label="Max", linewidth=1.5, linestyle="--")
        ax.plot(min_vals, c="red", label="Min", linewidth=1.5, linestyle="--")

        ax.set_title(f"Rewards (window = {window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()

        os.makedirs(dir, exist_ok=True)
        fname = f"{dir}/{self.config.slug()}.png"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        return fname

    def save_Q(self, dir):
        import pickle

        os.makedirs(dir, exist_ok=True)
        fname = f"{dir}/{self.config.slug()}.plk"
        with open(fname, "wb") as f:
            pickle.dump(self.Qbest, f)
        return fname

    def save_policy(self, dir):
        """Self-describing bundle: Q-table + config + discretizer name.

        Loading this cannot silently pick the wrong state grid, which is what
        made a coarse-trained table score like a random policy under the
        fine-grid lookup helper.
        """
        import pickle

        os.makedirs(dir, exist_ok=True)
        fname = f"{dir}/{self.config.slug()}.policy.pkl"
        with open(fname, "wb") as f:
            pickle.dump(
                {
                    "format": POLICY_FORMAT,
                    "Q": self.Qbest,
                    "config": self.config.to_dict(),
                    "discretizer": self.discretizer,
                    "best_avg": self.best_avg,
                    "at_epoch": self.at_epoch,
                    "seed": self.seed,
                    "total_epochs": self.total_epochs,
                },
                f,
            )
        return fname


POLICY_FORMAT = "cartpole-qlearning/1"


def save_policy(path, Q, config, discretizer="fine", **meta):
    """Write a self-describing policy bundle to an explicit path."""
    import pickle

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "format": POLICY_FORMAT,
        "Q": Q,
        "config": config.to_dict() if isinstance(config, Config) else dict(config),
        "discretizer": discretizer,
    }
    payload.update(meta)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_policy(path):
    """Load a policy bundle (or a bare Q-table pickle, with a warning).

    Returns ``(Q, config_or_None, discretizer_name)``.
    """
    import pickle

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and obj.get("format") == POLICY_FORMAT:
        cfg = Config.from_dict(obj["config"]) if obj.get("config") else None
        return obj["Q"], cfg, obj["discretizer"]

    # Legacy bare Q-table: we cannot know the grid it was built on.
    import warnings

    warnings.warn(
        f"{path} is a bare Q-table with no discretizer metadata; assuming 'fine'. "
        "If it was trained with discretize_bins ('coarse'), pass that explicitly "
        "or the lookup will miss every state and act randomly.",
        stacklevel=2,
    )
    return obj, None, "fine"
