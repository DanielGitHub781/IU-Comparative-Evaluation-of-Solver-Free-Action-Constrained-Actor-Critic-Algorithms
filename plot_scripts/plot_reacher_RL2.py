import os
import numpy as np
import matplotlib.pyplot as plt


def load_evaluations(filename: str):
    data = np.load(filename)
    return data["timesteps"], data["results"]  # results: (n_eval, n_episodes_per_eval)


def load_evaluations_multi_seed(prefix: str, n_seeds: int):
    """
    prefix e.g.: '../logs/R+L2-SAlpha'
    expects: '../logs/R+L2-SAlpha-1/evaluations.npz', ..., -5
    """
    rows = []
    timesteps = None

    for seed in range(1, n_seeds + 1):
        eval_path = f"{prefix}-{seed}/evaluations.npz"
        if not os.path.isfile(eval_path):
            raise FileNotFoundError(f"Missing file: {eval_path}")
        ts, results = load_evaluations(eval_path)
        mean_per_eval = results.mean(axis=1)  # mean over episodes

        if timesteps is None:
            timesteps = ts
        else:
            L = min(len(timesteps), len(ts))
            timesteps = timesteps[:L]
            mean_per_eval = mean_per_eval[:L]
        rows.append(mean_per_eval)

    # align all seeds to shortest length
    min_len = min(len(r) for r in rows)
    rows = [r[:min_len] for r in rows]
    timesteps = timesteps[:min_len]
    matrix = np.vstack(rows)  # shape: (n_seeds, n_eval)
    return timesteps, matrix


def plot_reacher_RL2_SAlpha(
    base_logs_dir: str = "../logs",
    n_seeds: int = 5,
    save_path: str = "reacher_R+L2_SAlpha_5seeds.png",
):
    algo = "SAlpha"
    prefix = os.path.join(base_logs_dir, f"R+L2-{algo}")

    timesteps, returns_matrix = load_evaluations_multi_seed(prefix, n_seeds)
    mean_curve = returns_matrix.mean(axis=0)
    std_curve = returns_matrix.std(axis=0)

    plt.figure("Reacher R+L2 – SAlpha", figsize=(6, 4))

    plt.plot(timesteps, mean_curve, label="SAlpha (SAC + alpha)")
    plt.fill_between(
        timesteps,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        label="±1 std",
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation return")
    plt.title("Reacher (R+L2) – SAlpha, 5 seeds")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_reacher_RL2_SAlpha()
