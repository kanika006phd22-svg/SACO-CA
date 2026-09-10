import numpy as np

from .config import SACOConfig
from .network import Network


def decode_control_nodes(position: np.ndarray, n_nodes: int, k: int) -> np.ndarray:
    """
    Convert continuous search-agent positions into exactly k unique
    discrete sensor-node indices.

    Mapping:
      1. round continuous coordinates to node indices;
      2. clip to the feasible range;
      3. remove duplicates;
      4. fill missing nodes using unused indices.

    This makes the continuous optimizer compatible with the discrete CN
    selection problem described in the supplied pseudocode.
    """
    raw = np.rint(position).astype(int)
    raw = np.clip(raw, 0, n_nodes - 1)

    selected = []
    used = set()

    for node in raw:
        if int(node) not in used:
            selected.append(int(node))
            used.add(int(node))
        if len(selected) == k:
            break

    if len(selected) < k:
        for node in np.argsort(-np.bincount(raw, minlength=n_nodes)):
            node = int(node)
            if node not in used:
                selected.append(node)
                used.add(node)
            if len(selected) == k:
                break

    if len(selected) < k:
        for node in range(n_nodes):
            if node not in used:
                selected.append(node)
                used.add(node)
            if len(selected) == k:
                break

    return np.asarray(selected, dtype=int)


def evaluate_fitness(
    position: np.ndarray,
    network: Network,
    cfg: SACOConfig,
) -> tuple[float, dict]:
    """
    Implements Equation (2):

    F(CH_i) =
        w1 * (1 - Ei/Emax)
      + w2 * (Di/Dmax)
      + w3 * (CBi/CBmax)
      + w4 * (1 - Qi/Qmax)

    Lower fitness is better.
    """
    cn = decode_control_nodes(position, cfg.n_nodes, cfg.n_control_nodes)

    energy = network.residual_energy
    emax = max(float(np.max(energy)), 1e-12)
    energy_term = float(np.mean(1.0 - energy[cn] / emax))

    # Assign each sensor to its nearest CN.
    sensor_to_cn_dist = np.linalg.norm(
        network.positions[:, None, :] - network.positions[cn][None, :, :],
        axis=2,
    )
    nearest_cn = np.argmin(sensor_to_cn_dist, axis=1)
    min_distances = sensor_to_cn_dist[np.arange(cfg.n_nodes), nearest_cn]

    d_i = float(np.mean(min_distances))
    d_max = max(float(np.max(sensor_to_cn_dist)), 1e-12)
    distance_term = d_i / d_max

    # Cluster balance factor:
    # CB = (1/K) * sum_k (N_k - Nbar)^2
    cluster_sizes = np.bincount(
        nearest_cn,
        minlength=cfg.n_control_nodes,
    ).astype(float)
    n_bar = cfg.n_nodes / cfg.n_control_nodes
    cb_i = float(np.mean((cluster_sizes - n_bar) ** 2))

    # Maximum possible variance-like reference for normalization.
    # This is a stable implementation-level CBmax.
    cb_max = max(
        ((cfg.n_nodes - n_bar) ** 2
         + (cfg.n_control_nodes - 1) * (n_bar ** 2))
        / cfg.n_control_nodes,
        1e-12,
    )
    cluster_balance_term = min(cb_i / cb_max, 1.0)

    # Average FSO quality between each CN and its assigned sensors.
    q_values = network.fso_quality[cn[nearest_cn], np.arange(cfg.n_nodes)]
    q_i = float(np.mean(q_values))
    q_max = max(float(np.max(network.fso_quality)), 1e-12)
    fso_term = float(1.0 - q_i / q_max)

    fitness = (
        cfg.w1 * energy_term
        + cfg.w2 * distance_term
        + cfg.w3 * cluster_balance_term
        + cfg.w4 * fso_term
    )

    details = {
        "control_nodes": cn,
        "energy_term": energy_term,
        "distance_term": distance_term,
        "cluster_balance_term": cluster_balance_term,
        "fso_term": fso_term,
        "average_distance": d_i,
        "cluster_balance": cb_i,
        "average_fso_quality": q_i,
        "fitness": float(fitness),
    }

    return float(fitness), details
