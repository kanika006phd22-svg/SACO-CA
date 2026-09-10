import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saco_ca.config import SACOConfig
from saco_ca.network import create_network
from saco_ca.optimizer import SACOCA


def main():
    parser = argparse.ArgumentParser(
        description="Run the SACO-CA Self-Adaptive Cheetah Optimization-Based Clustering algorithm."
    )
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--cn", type=int, default=40)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cfg = SACOConfig(
        n_nodes=args.nodes,
        n_control_nodes=args.cn,
        population_size=args.population,
        max_iterations=args.iterations,
        seed=args.seed,
    )

    network = create_network(cfg)
    optimizer = SACOCA(network, cfg)
    result = optimizer.run()

    # Save optimized CNs.
    cn_df = pd.DataFrame({
        "control_node": result["control_nodes"] + 1,  # 1-based manuscript indexing
    })
    cn_df.to_csv(output / "optimized_control_nodes.csv", index=False)

    # Save fitness components.
    summary = pd.DataFrame([{
        "fitness": result["fitness"],
        "energy_term": result["details"]["energy_term"],
        "distance_term": result["details"]["distance_term"],
        "cluster_balance_term": result["details"]["cluster_balance_term"],
        "fso_term": result["details"]["fso_term"],
        "average_distance": result["details"]["average_distance"],
        "cluster_balance": result["details"]["cluster_balance"],
        "average_fso_quality": result["details"]["average_fso_quality"],
    }])
    summary.to_csv(output / "fitness_summary.csv", index=False)

    # Save convergence.
    convergence_df = pd.DataFrame({
        "iteration": np.arange(1, len(result["convergence"]) + 1),
        "best_fitness": result["convergence"],
    })
    convergence_df.to_csv(output / "convergence.csv", index=False)

    # Publication-friendly convergence figure.
    plt.figure(figsize=(8, 5))
    plt.plot(
        convergence_df["iteration"],
        convergence_df["best_fitness"],
        linewidth=2,
    )
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Fitness")
    plt.title("SACO-CA Convergence")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "saco_ca_convergence.png", dpi=300)
    plt.close()

    print("\nSACO-CA completed")
    print("-" * 50)
    print(f"Nodes                : {cfg.n_nodes}")
    print(f"Control Nodes (K)    : {cfg.n_control_nodes}")
    print(f"Population Size      : {cfg.population_size}")
    print(f"Iterations           : {cfg.max_iterations}")
    print(f"Seed                 : {cfg.seed}")
    print(f"Best Fitness         : {result['fitness']:.8f}")
    print(
        "Optimized CNs (1-based): "
        + ", ".join(map(str, result["control_nodes"] + 1))
    )
    print(f"Results saved to     : {output.resolve()}")


if __name__ == "__main__":
    main()
