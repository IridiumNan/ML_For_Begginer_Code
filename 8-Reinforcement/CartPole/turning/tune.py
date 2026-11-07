"""Bayesian hyperparameter search for tabular CartPole Q-learning.

Three stages, all driven by a process pool (default 6 workers):

  calibrate  measure run-to-run noise for a few fixed configs, so the number of
             seeds per trial can be chosen from evidence rather than taste
  search     Optuna (TPE) over learning-rate strategy + its parameters, gamma,
             epsilon, terminal punish and the discretization scheme
  validate   re-rank the best few configs with more seeds and a larger budget

Usage
    python tune.py calibrate --workers 6
    python tune.py search    --workers 6 --trials 80 --episodes 4000
    python tune.py validate  --workers 6 --top 8 --seeds 5 --episodes 12000

Each worker runs one training job at a time (the Q-learning loop is pure
Python, so processes -- not threads -- are what actually parallelize it).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "mpl")
)

import numpy as np

from model import ALPHA_STRATEGIES, Config, DEFAULT_ALPHA_PARAMS, save_policy
from utils import DISCRETIZERS

DISCRETIZERS = dict(DISCRETIZERS)

#: Far above every training seed range (trial.number * 1000 + episodes), so an
#: evaluation never starts from an initial state the policy was trained on.
EVAL_SEED_OFFSET = 20_000_000

OUT_DIR = "tuning"
_ENV = None


# ------------------------------------------------------------------ workers
def _worker_init():
    """Runs once per worker process."""
    global _ENV
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "mpl"),
    )
    from train import make_env

    _ENV = make_env()


def run_one(job):
    """Train once and greedily evaluate. Pure function of ``job``.

    ``job`` = {"params": {...}, "seed": int, "episodes": int, "eval_episodes": int}
    """
    global _ENV
    if _ENV is None:  # direct (non-pool) call
        _worker_init()

    from train import evaluate, train

    params = job["params"]
    seed = job["seed"]
    drz = DISCRETIZERS[params["discretizer"]]

    cfg = Config.build(
        name=f"s{seed}",
        strategy=params["strategy"],
        alpha_params=params["alpha_params"],
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        punish=params["punish"],
    )
    try:
        r = train(
            _ENV, cfg,
            total_epochs=job["episodes"],
            batch=50,
            seed=seed,
            discretizer=drz,
        )
        ev = evaluate(
            _ENV, r.Qbest, episodes=job.get("eval_episodes", 40),
            seed=EVAL_SEED_OFFSET, discretizer=drz,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}

    out = {
        "seed": seed,
        "greedy_mean": float(ev.mean()),
        "greedy_median": float(np.median(ev)),
        "greedy_min": float(ev.min()),
        "greedy_max": float(ev.max()),
        "train_best_avg": float(r.best_avg),
        "train_final100": r.final_window(100),
        "best_at_epoch": r.at_epoch,
        "table_size": len(r.Qbest),
        "elapsed": r.elapsed,
    }
    if job.get("keep_table"):
        out["Q"] = r.Qbest
    return out


def _aggregate(results, key="greedy_mean"):
    ok = [r for r in results if "error" not in r]
    if not ok:
        return None
    vals = np.array([r[key] for r in ok], dtype=float)
    # Q-tables are carried only for the final artifact; keep them out of summaries
    # so they never end up in JSON or in every printed row.
    slim = [{k: v for k, v in r.items() if k != "Q"} for r in results]
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": len(vals),
        "runs": slim,
    }


def _fmt_params(p):
    ap = " ".join(f"{k}={v:.4g}" for k, v in sorted(p["alpha_params"].items()))
    return (
        f"{p['strategy']:<9s}[{ap}] g={p['gamma']:.3f} e={p['epsilon']:.3f} "
        f"p={p['punish']:.1f} disc={p['discretizer']}"
    )


# --------------------------------------------------------------- search space
def suggest_params(trial, episodes):
    """Optuna suggestion -> the flat parameter dict the workers consume."""
    strategy = trial.suggest_categorical("strategy", list(ALPHA_STRATEGIES))

    if strategy == "constant":
        ap = {"alpha": trial.suggest_float("constant_alpha", 0.01, 1.0, log=True)}
    elif strategy == "linear":
        a0 = trial.suggest_float("linear_a0", 0.05, 1.0, log=True)
        a1 = trial.suggest_float("linear_a1", 1e-3, a0, log=True)
        horizon = trial.suggest_int("linear_horizon", 200, min(20000, max(400, 4 * episodes)), log=True)
        ap = {"a0": a0, "a1": a1, "horizon": horizon}
    elif strategy == "exp":
        a0 = trial.suggest_float("exp_a0", 0.05, 1.0, log=True)
        # half-life in episodes is far easier for TPE to model than a decay rate
        half_life = trial.suggest_float("exp_half_life", 100.0, min(20000.0, max(200.0, 5.0 * episodes)), log=True)
        ap = {"a0": a0, "decay": 0.5 ** (1.0 / half_life), "floor": 1e-3}
    elif strategy == "invsqrt":
        a0 = trial.suggest_float("invsqrt_a0", 0.05, 1.0, log=True)
        scale = trial.suggest_float("invsqrt_scale", 10.0, min(20000.0, max(200.0, 5.0 * episodes)), log=True)
        ap = {"a0": a0, "scale": scale, "floor": 1e-3}
    elif strategy == "step":
        a0 = trial.suggest_float("step_a0", 0.05, 1.0, log=True)
        factor = trial.suggest_float("step_factor", 0.3, 0.95)
        every = trial.suggest_int("step_every", 100, min(10000, max(200, episodes // 2)), log=True)
        ap = {"a0": a0, "factor": factor, "every": every, "floor": 1e-3}
    else:  # pragma: no cover
        raise ValueError(strategy)

    return {
        "strategy": strategy,
        "alpha_params": ap,
        "gamma": trial.suggest_float("gamma", 0.80, 0.999),
        "epsilon": trial.suggest_float("epsilon", 0.005, 0.30, log=True),
        "punish": trial.suggest_float("punish", -30.0, -1.0),
        "discretizer": trial.suggest_categorical("discretizer", list(DISCRETIZERS)),
    }


# -------------------------------------------------------------------- stages
def stage_calibrate(args):
    """Same config, several seeds -> how much of a score gap is just noise."""
    from train import make_env  # noqa: F401  (ensures import cost is paid once)

    configs = [
        {"strategy": "constant", "alpha_params": {"alpha": 0.5}, "gamma": 0.95,
         "epsilon": 0.05, "punish": -10.0, "discretizer": "fine"},
        {"strategy": "linear", "alpha_params": {"a0": 0.5, "a1": 0.02, "horizon": 4000},
         "gamma": 0.97, "epsilon": 0.05, "punish": -10.0, "discretizer": "fine"},
        {"strategy": "exp", "alpha_params": {"a0": 0.5, "decay": 0.5 ** (1 / 1500), "floor": 1e-3},
         "gamma": 0.99, "epsilon": 0.02, "punish": -10.0, "discretizer": "coarse"},
    ]
    jobs = [
        {"params": p, "seed": s, "episodes": args.episodes, "eval_episodes": 40}
        for p in configs for s in range(args.seeds)
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        results = list(ex.map(run_one, jobs))
    dt = time.time() - t0

    print(f"\ncalibration: {len(jobs)} runs, {args.workers} workers, "
          f"{args.episodes} episodes, wall {dt:.1f}s")
    k = args.seeds
    for i, p in enumerate(configs):
        agg = _aggregate(results[i * k:(i + 1) * k])
        print(f"\n  {_fmt_params(p)}")
        if agg is None:
            print("    all runs failed")
            continue
        print(f"    greedy mean {agg['mean']:7.2f}  sd {agg['sd']:6.2f} "
              f"range [{agg['min']:.0f}, {agg['max']:.0f}]")
        print("    per-seed      " + "  ".join(f"{r['greedy_mean']:.0f}" for r in agg["runs"]))

    all_sd = []
    for i in range(len(configs)):
        agg = _aggregate(results[i * k:(i + 1) * k])
        if agg:
            all_sd.append(agg["sd"])
    if all_sd:
        print(f"\n  median within-config seed sd: {np.median(all_sd):.2f} reward")
        print("  => a config must beat another by ~{:.1f} to be a real difference at 1 seed".format(
            float(np.median(all_sd))))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/calibration.json", "w") as f:
        json.dump({"episodes": args.episodes, "seeds": args.seeds,
                   "wall": dt, "jobs": jobs, "results": results}, f, indent=2)


def stage_search(args):
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(OUT_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{OUT_DIR}/{args.study}.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=args.startup,
            multivariate=True,
            seed=args.seed,
        ),
    )
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if done:
        print(f"resuming {args.study}: {len(done)} completed trials, best {study.best_value:.2f}")

    budget = args.trials - len([t for t in study.trials
                                if t.state in (TrialState.COMPLETE, TrialState.RUNNING)])
    if budget <= 0:
        print("nothing to do")
        return
    print(f"running {budget} trials: {args.episodes} episodes, {args.seeds} seed(s), "
          f"{args.workers} workers\n")

    t_start = time.time()
    scored = []

    def build_jobs(trial):
        params = suggest_params(trial, args.episodes)
        return params, [
            {"params": params, "seed": trial.number * 1000 + s,
             "episodes": args.episodes, "eval_episodes": args.eval_episodes}
            for s in range(args.seeds)
        ]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        pending = {}       # future -> (trial, params, n_seeds_expected)
        inflight_jobs = {}  # trial.number -> list of results

        def submit_trial():
            trial = study.ask()
            params, jobs = build_jobs(trial)
            inflight_jobs[trial.number] = []
            for j in jobs:
                fut = ex.submit(run_one, j)
                pending[fut] = (trial, params, len(jobs))

        for _ in range(min(args.workers, budget)):
            submit_trial()

        finished = 0
        while pending:
            done_futs, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                trial, params, expected = pending.pop(fut)
                try:
                    inflight_jobs[trial.number].append(fut.result())
                except Exception as exc:
                    inflight_jobs[trial.number].append({"error": f"{type(exc).__name__}: {exc}"})

                # only finalize once every seed of this trial has reported
                if len(inflight_jobs[trial.number]) < expected:
                    continue

                agg = _aggregate(inflight_jobs.pop(trial.number))
                if agg is None:
                    study.tell(trial, state=TrialState.FAIL)
                    print(f"  trial {trial.number:3d} FAILED")
                else:
                    # Unbiased multi-seed mean. The sample sd of 3 seeds is far too
                    # unreliable to penalise with (it is noisy and biased low), so it
                    # is recorded but the honest ranking happens in the validate stage.
                    score = agg["mean"]
                    for k, v in {
                        "greedy_mean": agg["mean"],
                        "greedy_sd": agg["sd"],
                        "train_best_avg": float(np.mean([r["train_best_avg"] for r in agg["runs"]])),
                        "table_size": int(np.mean([r["table_size"] for r in agg["runs"]])),
                        "params": json.dumps(params),
                    }.items():
                        trial.set_user_attr(k, v)
                    study.tell(trial, score, state=TrialState.COMPLETE)
                    scored.append((score, trial.number, params, agg))
                    print(f"  trial {trial.number:3d}  score {score:7.2f} "
                          f"(mean {agg['mean']:7.2f} sd {agg['sd']:5.2f})  {_fmt_params(params)}")
                finished += 1
                if finished < budget:
                    submit_trial()

    study.trials_dataframe().to_csv(f"{OUT_DIR}/{args.study}_trials.csv", index=False)
    ranked = sorted(scored, key=lambda x: -x[0])[:args.report]
    print(f"\nsearch done in {time.time() - t_start:.1f}s. best score {study.best_value:.2f}")
    print(f"\ntop {len(ranked)}:")
    for i, (s, num, p, agg) in enumerate(ranked, 1):
        print(f"  {i:2d}. {s:7.2f}  trial {num:3d}  {_fmt_params(p)}")
        print(f"      greedy mean {agg['mean']:7.2f}  sd {agg['sd']:5.2f}  "
              f"train {np.mean([r['train_best_avg'] for r in agg['runs']]):7.2f}")

    with open(f"{OUT_DIR}/{args.study}_top.json", "w") as f:
        json.dump([{"score": s, "trial": n, "params": p, "agg": a} for s, n, p, a in ranked],
                  f, indent=2)


def stage_validate(args):
    """Re-run the top configs from the study with more seeds and a bigger budget."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    top_path = f"{OUT_DIR}/{args.study}_top.json"
    if not os.path.exists(top_path):
        sys.exit(f"missing {top_path}; run the search stage first")
    with open(top_path) as f:
        top = json.load(f)[:args.top]

    jobs, owners = [], []
    for i, entry in enumerate(top):
        for s in range(args.seeds):
            jobs.append({"params": entry["params"], "seed": 7000 + i * 100 + s,
                         "episodes": args.episodes, "eval_episodes": args.eval_episodes,
                         "keep_table": s == 0})
            owners.append(i)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        results = list(ex.map(run_one, jobs))
    dt = time.time() - t0

    grouped = {i: [] for i in range(len(top))}
    for owner, r in zip(owners, results):
        grouped[owner].append(r)

    rows = []
    for i, entry in enumerate(top):
        agg = _aggregate(grouped[i])
        if agg is None:
            continue
        rows.append({
            "rank": None,
            "params": entry["params"],
            "greedy_mean": agg["mean"],
            "greedy_sd": agg["sd"],
            "greedy_min": agg["min"],
            "greedy_max": agg["max"],
            "train_best_avg": float(np.mean([r["train_best_avg"] for r in agg["runs"]])),
            "table_size": int(np.mean([r["table_size"] for r in agg["runs"]])),
            "runs": agg["runs"],
            "Q": next((r["Q"] for r in grouped[i] if "Q" in r), None),
        })
    rows.sort(key=lambda r: -r["greedy_mean"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    print(f"\nvalidation: top {len(top)} configs, {args.seeds} seeds each, "
          f"{args.episodes} episodes, wall {dt:.1f}s")
    for r in rows:
        print(f"\n  #{r['rank']}  greedy mean {r['greedy_mean']:7.2f} sd {r['greedy_sd']:5.2f} "
              f"[{r['greedy_min']:.0f}, {r['greedy_max']:.0f}]  table {r['table_size']}")
        print(f"      {_fmt_params(r['params'])}")

    with open(f"{OUT_DIR}/{args.study}_validate.json", "w") as f:
        json.dump({"episodes": args.episodes, "seeds": args.seeds, "wall": dt,
                   "rows": [{k: v for k, v in r.items() if k != "Q"} for r in rows]},
                  f, indent=2)
    if rows:
        best = rows[0]
        print(f"\nbest after validation: {_fmt_params(best['params'])}")
        print(f"  greedy mean {best['greedy_mean']:.2f} (sd {best['greedy_sd']:.2f})")
        print(f"  written to {OUT_DIR}/{args.study}_validate.json")

        # Publish a runnable artifact: the bundle records its own state grid, so
        # play.py can never look the table up on the wrong discretization.
        if best.get("Q"):
            p = best["params"]
            cfg = Config.build(
                name="best", strategy=p["strategy"], alpha_params=p["alpha_params"],
                gamma=p["gamma"], epsilon=p["epsilon"], punish=p["punish"],
            )
            path = save_policy(
                f"{OUT_DIR}/{args.study}_best.policy.pkl", best["Q"], cfg,
                discretizer=p["discretizer"],
                greedy_mean=best["greedy_mean"], greedy_sd=best["greedy_sd"],
                episodes=args.episodes, seeds=args.seeds,
            )
            print(f"  policy  {path}")
            print(f"  play    python play.py {path} --episodes 20")


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp):
        sp.add_argument("--workers", type=int, default=6)
        sp.add_argument("--episodes", type=int, default=4000)
        sp.add_argument("--eval-episodes", type=int, default=40)
        sp.add_argument("--seed", type=int, default=0)

    cal = sub.add_parser("calibrate")
    common(cal)
    cal.add_argument("--seeds", type=int, default=5)
    cal.set_defaults(func=stage_calibrate)

    sr = sub.add_parser("search")
    common(sr)
    sr.add_argument("--trials", type=int, default=80)
    sr.add_argument("--seeds", type=int, default=1, help="seeds per trial (objective averages them)")
    sr.add_argument("--startup", type=int, default=15, help="random trials before TPE")
    sr.add_argument("--study", default="cartpole")
    sr.add_argument("--report", type=int, default=10)
    sr.set_defaults(func=stage_search)

    va = sub.add_parser("validate")
    common(va)
    va.add_argument("--top", type=int, default=8)
    va.add_argument("--seeds", type=int, default=5)
    va.add_argument("--study", default="cartpole")
    va.set_defaults(func=stage_validate)

    args = p.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
