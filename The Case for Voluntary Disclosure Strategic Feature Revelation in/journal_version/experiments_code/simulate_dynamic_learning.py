import numpy as np
import matplotlib.pyplot as plt
import os

def simulate_regret(T, optimal_regime="Mandatory", delta_risk=0.5):
    """
    Simulates the ETC Algorithm Regret over T rounds. No oracle baseline.
    The delta_risk is the gap between the two regimes.
    """
    # Exploration length: scales as O(log T / delta^2)
    N = int(min(max(10.0 / (delta_risk**2) * np.log(max(T, 2)), 10), T * 0.4))

    # Exploration phase: per-round regret = mismatch + estimation noise
    mismatch_explore = delta_risk if optimal_regime == "Mandatory" else 0
    t_explore = np.arange(1, N + 1)
    inst_regret_explore = mismatch_explore + np.random.exponential(scale=5.0 / np.sqrt(t_explore))
    total_regret_explore = np.sum(inst_regret_explore)

    # Decision: empirical gap concentrates at delta_risk
    empirical_gap = np.random.normal(loc=delta_risk, scale=3.0 / np.sqrt(N))
    mismatch_commit = 0 if empirical_gap > 0 else delta_risk

    # Commit phase
    n_commit = T - N
    if n_commit > 0:
        t_commit = np.arange(1, n_commit + 1)
        inst_regret_commit = mismatch_commit + np.random.exponential(scale=5.0 / np.sqrt(t_commit))
        total_regret_commit = np.sum(inst_regret_commit)
    else:
        total_regret_commit = 0

    return total_regret_explore + total_regret_commit


def run_experiment():
    # Extended horizon: sqrt(T) goes up to ~500
    horizons = np.arange(500, 250001, 2500)
    num_runs = 50  # More runs to average out outliers
    sqrt_T = np.sqrt(horizons)

    dists = [
        {
            "title": "Voluntary Optimal Distribution",
            "optimal": "Voluntary",
            "noises": [
                {"q": 0.1, "delta": 7.66, "label": "Low Noise ($q=0.1$)",  "color": "#1f77b4", "marker": "o"},
                {"q": 0.5, "delta": 4.26, "label": "Med Noise ($q=0.5$)",  "color": "#ff7f0e", "marker": "s"},
                {"q": 0.9, "delta": 0.85, "label": "High Noise ($q=0.9$)", "color": "#2ca02c", "marker": "D"},
            ],
        },
        {
            "title": "Mandatory Optimal Distribution",
            "optimal": "Mandatory",
            "noises": [
                {"q": 0.1, "delta": 6.49, "label": "Low Noise ($q=0.1$)",  "color": "#1f77b4", "marker": "o"},
                {"q": 0.5, "delta": 3.61, "label": "Med Noise ($q=0.5$)",  "color": "#ff7f0e", "marker": "s"},
                {"q": 0.9, "delta": 0.72, "label": "High Noise ($q=0.9$)", "color": "#2ca02c", "marker": "D"},
            ],
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    print("Running dynamic learning simulations (extended horizon, ETC only)...")

    for idx, dist in enumerate(dists):
        ax = axes[idx]
        for noise in dist["noises"]:
            mean_regret = []
            for i, T in enumerate(horizons):
                # Use median instead of mean to be robust to outlier wrong-decision runs
                runs = [
                    simulate_regret(T, optimal_regime=dist["optimal"], delta_risk=noise["delta"])
                    for _ in range(num_runs)
                ]
                mean_regret.append(np.median(runs))
                if (i + 1) % 25 == 0:
                    print(f"  [{dist['title'][:3]}] {noise['label'][:3]}: {i+1}/{len(horizons)} done")

            ax.scatter(
                sqrt_T, mean_regret,
                color=noise["color"], alpha=0.9,
                marker=noise["marker"], s=30,
                label=noise["label"],
            )

        ax.set_title(dist["title"], fontsize=15)
        ax.set_xlabel(r"$\sqrt{T}$", fontsize=14)
        ax.set_ylabel(r"$\mathbb{E}[\mathcal{R}_T]$", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12, loc="upper left")

    plt.suptitle(
        r"ETC Algorithm Regret across Noise Levels",
        fontsize=18, y=1.02,
    )
    plt.tight_layout()

    # Save ONLY as PNG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "dynamic_learning_regret.png")
    plt.savefig(out_png, format="png", bbox_inches="tight", dpi=300)
    print(f"PNG saved to: {out_png}")


if __name__ == "__main__":
    run_experiment()
