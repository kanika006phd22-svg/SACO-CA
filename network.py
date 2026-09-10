from dataclasses import dataclass
import numpy as np

from .config import SACOConfig


@dataclass
class Network:
    positions: np.ndarray
    residual_energy: np.ndarray
    fso_quality: np.ndarray


def create_network(cfg: SACOConfig) -> Network:
    """
    Create a reproducible underwater optical sensor network.

    Nodes are represented by 2-D coordinates. The FSO quality matrix is
    calculated from distance, attenuation, alignment loss and turbidity
    perturbation. Values are clipped to [0, 1].
    """
    rng = np.random.default_rng(cfg.seed)

    positions = rng.uniform(
        low=[0.0, 0.0],
        high=[cfg.field_x, cfg.field_y],
        size=(cfg.n_nodes, 2),
    )

    residual_energy = np.full(cfg.n_nodes, cfg.initial_energy, dtype=float)

    # Pairwise Euclidean distance
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # A compact normalized FSO quality model.
    # q_ij = exp(-attenuation * d) * alignment * turbidity
    alignment = np.clip(
        1.0 - np.abs(rng.normal(0.0, cfg.alignment_std, distances.shape)),
        0.0,
        1.0,
    )
    turbidity = np.clip(
        np.exp(-np.abs(rng.normal(0.0, cfg.turbidity_std, distances.shape))),
        0.0,
        1.0,
    )

    fso_quality = (
        np.exp(-cfg.attenuation_coefficient * distances)
        * alignment
        * turbidity
    )
    np.fill_diagonal(fso_quality, 1.0)
    fso_quality = np.clip(fso_quality, 0.0, 1.0)

    return Network(
        positions=positions,
        residual_energy=residual_energy,
        fso_quality=fso_quality,
    )
