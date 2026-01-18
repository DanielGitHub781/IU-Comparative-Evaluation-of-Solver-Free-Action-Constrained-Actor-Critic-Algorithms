import argparse
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Data structures
@dataclass
class RunInfo:
    run_dir: Path
    prob_id: str
    algo_id: str
    seed: Optional[int]
    env: Optional[str]


# Helpers
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def discover_runs(logs_dir: Path) -> List[RunInfo]:
    runs = []
    for d in logs_dir.iterdir():
        if not d.is_dir():
            continue

        args_file = d / "commandline_args.txt"
        eval_file = d / "evaluations.npz"
        if not args_file.exists() or not eval_file.exists():
            continue

        args = read_json(args_file)
        prob_id = args.get("prob_id", "")
        algo_id = args.get("algo_id", "")
        seed = args.get("seed", None)
        env = args.get("env", None)

        # fallback: infer from folder name
        if not prob_id or not algo_id:
            parts = d.name.split("-")
            if len(parts) >= 3 and parts[-1].isdigit():
                seed = seed or int(parts[-1])
                algo_id = algo_id or parts[-2]
                prob_id = prob_id or "-".join(parts[:-2])

        if prob_id and algo_id:
            try:
                seed = int(seed) if seed is not None else None
            except Exception:
                seed = None
            runs.append(RunInfo(d, prob_id, algo_id, seed, env))

    return runs


def load_evaluations(path: Path):
    data = np.load(path)
    return data["timesteps"], data["results"]


LOG_TIME_RE = re.compile(r"\|\s*time_elapsed\s*\|\s*([0-9\.]+)")
LOG_TS_RE   = re.compile(r"\|\s*total_timesteps\s*\|\s*([0-9\.]+)")


def parse_runtime(log_path: Path):
    if not log_path.exists():
        return np.nan, np.nan
    txt = log_path.read_text(errors="ignore")
    times = LOG_TIME_RE.findall(txt)
    steps = LOG_TS_RE.findall(txt)
    t = float(times[-1]) if times else np.nan
    s = float(steps[-1]) if steps else np.nan
    return t, s


def auc(x, y):
    return np.trapz(y, x) if len(x) > 1 else np.nan


def time_to_threshold(timesteps, curve, threshold):
    idx = np.where(curve >= threshold)[0]
    return timesteps[idx[0]] if len(idx) > 0 else np.nan


def mean_std_se(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(x)
    std = np.std(x, ddof=1) if len(x) > 1 else 0.0
    se = std / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return mean, std, se


def group_result_path_for_run(run_dir: Path) -> Optional[Path]:
    name = run_dir.name
    if "-" not in name:
        return None
    base = name.rsplit("-", 1)[0]  
    p = run_dir.parent / base / "result.npy"
    return p if p.exists() and p.is_file() else None


def load_final_50ep_for_seed(run_dir: Path, seed: Optional[int]) -> float:
    if seed is None:
        return np.nan
    p = group_result_path_for_run(run_dir)
    if p is None:
        return np.nan

    arr = np.load(p)
    if arr.ndim != 1:
        return np.nan

    idx = int(seed) - 1
    if idx < 0 or idx >= len(arr):
        return np.nan
    return float(arr[idx])
 

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--out_dir", default="metrics_out")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Reward threshold for learning speed metric")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    runs = discover_runs(logs_dir)
    if not runs:
        raise RuntimeError("No valid runs found.")

    per_run = []
    groups: Dict[Tuple[str, str], List[dict]] = {}

    # Per-run metrics
    for r in runs:
        timesteps, results = load_evaluations(r.run_dir / "evaluations.npz")
        curve = results.mean(axis=1)

        final_return = curve[-1]
        auc_val = auc(timesteps, curve)

        t_elapsed, total_steps = parse_runtime(r.run_dir / "log.txt")
        sec_per_1k = t_elapsed / (total_steps / 1000) if total_steps > 0 else np.nan

        ttt = time_to_threshold(timesteps, curve, args.threshold) \
              if args.threshold is not None else np.nan

        # Best-model final eval (50 episodes) from result.npy
        final_return_50ep = load_final_50ep_for_seed(r.run_dir, r.seed)

        per_run.append(dict(
            run=str(r.run_dir),
            prob_id=r.prob_id,
            algo_id=r.algo_id,
            seed=r.seed,
            final_return=final_return,                
            final_return_50ep=final_return_50ep,      
            auc=auc_val,
            time_to_threshold=ttt,
            time_elapsed_s=t_elapsed,
            total_timesteps=total_steps,
            sec_per_1k_steps=sec_per_1k,
        ))

        groups.setdefault((r.prob_id, r.algo_id), []).append(
            dict(t=timesteps, y=curve)
        )

    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(out_dir / "metrics_per_run.csv", index=False)

    # Aggregated metrics + plots
    summary = []

    for (prob_id, algo_id), curves in groups.items():
        sub = per_run_df[
            (per_run_df.prob_id == prob_id) &
            (per_run_df.algo_id == algo_id)
        ]

        final_mean, final_std, final_se = mean_std_se(sub.final_return.values)
        final50_mean, final50_std, final50_se = mean_std_se(sub.final_return_50ep.values)

        auc_mean, auc_std, auc_se = mean_std_se(sub.auc.values)
        rt_mean, rt_std, rt_se = mean_std_se(sub.time_elapsed_s.values)
        s1k_mean, s1k_std, s1k_se = mean_std_se(sub.sec_per_1k_steps.values)

        ttt_mean, ttt_std, ttt_se = mean_std_se(sub.time_to_threshold.values) \
            if args.threshold is not None else (np.nan, np.nan, np.nan)

        # Training budget
        total_ts_median = float(np.nanmedian(sub.total_timesteps.values)) if len(sub) else np.nan

        summary.append(dict(
            prob_id=prob_id,
            algo_id=algo_id,
            n_seeds=len(sub),

            # last evaluation point during training
            final_mean=final_mean,
            final_std=final_std,
            final_se=final_se,

            # best-model 50-episode evaluation
            final50_mean=final50_mean,
            final50_std=final50_std,
            final50_se=final50_se,

            # learning speed / overall
            auc_mean=auc_mean,
            auc_std=auc_std,
            auc_se=auc_se,
            time_to_threshold_mean=ttt_mean,
            time_to_threshold_std=ttt_std,

            # runtime
            runtime_s_mean=rt_mean,
            runtime_s_std=rt_std,
            sec_per_1k_mean=s1k_mean,
            sec_per_1k_std=s1k_std,

            # training budget
            total_timesteps_median=total_ts_median,
        ))

        # learning curve plot
        L = min(len(c["y"]) for c in curves)
        Y = np.vstack([c["y"][:L] for c in curves])
        t = curves[0]["t"][:L]

        mean_curve = Y.mean(axis=0)
        se_curve = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean_curve)

        plt.figure(figsize=(6, 4))
        plt.plot(t, mean_curve)
        plt.fill_between(t, mean_curve - se_curve, mean_curve + se_curve, alpha=0.3)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean evaluation return")
        plt.title(f"{prob_id} – {algo_id}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{prob_id}__{algo_id}.png", dpi=200)
        plt.close()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    print("Saved:")
    print(" - metrics_per_run.csv")
    print(" - metrics_summary.csv")
    print(" - plots/*.png")


if __name__ == "__main__":
    main()

