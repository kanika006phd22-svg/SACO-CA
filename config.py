from dataclasses import dataclass


@dataclass
class SACOConfig:
    # Network
    n_nodes: int = 500
    n_control_nodes: int = 40
    field_x: float = 1000.0
    field_y: float = 1000.0
    initial_energy: float = 2.0

    # Optimizer
    population_size: int = 30
    max_iterations: int = 300
    dimensions: int | None = None
    seed: int = 42

    # Fitness weights from the proposed formulation
    w1: float = 0.35  # energy
    w2: float = 0.25  # distance
    w3: float = 0.20  # cluster balance
    w4: float = 0.20  # FSO quality

    # FSO/link model
    max_fso_distance: float = 1000.0
    attenuation_coefficient: float = 0.001
    alignment_std: float = 0.03
    turbidity_std: float = 0.02

    # Cheetah-inspired search parameters
    hunting_time_scale: float = 60.0
    sit_wait_probability: float = 0.30
    attack_probability: float = 0.60
    leave_return_probability: float = 0.20
    exploration_scale: float = 0.15

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = self.n_control_nodes

        if self.n_control_nodes > self.n_nodes:
            raise ValueError("n_control_nodes cannot exceed n_nodes.")

        if abs(self.w1 + self.w2 + self.w3 + self.w4 - 1.0) > 1e-9:
            raise ValueError("Fitness weights must sum to 1.")
