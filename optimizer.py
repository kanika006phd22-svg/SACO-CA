import numpy as np

from .config import SACOConfig
from .fitness import evaluate_fitness
from .network import Network


class SACOCA:
    """
    Self-Adaptive Cheetah Optimization-Based Clustering.

    The implementation follows the supplied 40-line pseudocode at the
    algorithmic level. The original manuscript notation for SF, TR, TU,
    and the exact position-update equations is not fully specified in the
    supplied images; therefore, these update rules are implemented as
    configurable cheetah-inspired numerical operators.
    """

    def __init__(self, network: Network, cfg: SACOConfig):
        self.network = network
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.lower = 0.0
        self.upper = float(cfg.n_nodes - 1)

        # Line 1: initialize positions of all search agents.
        self.positions = self.rng.uniform(
            self.lower,
            self.upper,
            size=(cfg.population_size, cfg.dimensions),
        )

        self.fitness = np.full(cfg.population_size, np.inf)
        self.details = [None] * cfg.population_size

        self.best_position = None
        self.best_fitness = np.inf
        self.best_details = None

        self.convergence = []

    def _evaluate_population(self):
        for i in range(self.cfg.population_size):
            fit, details = evaluate_fitness(
                self.positions[i], self.network, self.cfg
            )
            self.fitness[i] = fit
            self.details[i] = details

            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_position = self.positions[i].copy()
                self.best_details = details.copy()

    def _adaptive_inertia(self, iteration: int) -> float:
        # Self-adaptive inertia: larger exploration at the beginning,
        # progressively more exploitation near the final iteration.
        progress = iteration / max(self.cfg.max_iterations - 1, 1)
        return 0.90 - 0.70 * progress

    def _searching_behaviour(self, x, leader, inertia):
        r = self.rng.uniform(0.0, 1.0, size=self.cfg.dimensions)
        exploration = (
            self.cfg.exploration_scale
            * (1.0 - r)
            * (self.upper - self.lower)
        )
        return (
            x
            + inertia * r * (leader - x)
            + self.rng.normal(0.0, 1.0, x.shape) * exploration
        )

    def _attacking_behaviour(self, x, leader, prey, iteration):
        # Exploitation around leader/prey.
        sf = self.rng.uniform(0.0, 1.0)
        tr = self.rng.uniform(0.0, 1.0)
        tu = self.rng.uniform(0.0, 1.0)

        direction = (
            sf * (leader - x)
            + tr * (prey - x)
            + tu * (leader - prey)
        )
        step = np.exp(-2.0 * iteration / max(self.cfg.max_iterations, 1))
        return x + step * direction + self.rng.normal(0, 0.05, x.shape)

    def _sit_and_wait_behaviour(self, x, leader):
        # Small local perturbation while remaining close to the leader.
        noise = self.rng.normal(
            0.0,
            self.cfg.exploration_scale * (self.upper - self.lower) * 0.05,
            size=x.shape,
        )
        return x + 0.25 * (leader - x) + noise

    def _leave_and_return(self, previous_position):
        # Preserve leader position in the event of a hunting-time trigger,
        # then restart exploration around a randomly generated position.
        new_position = self.rng.uniform(
            self.lower,
            self.upper,
            size=self.cfg.dimensions,
        )
        return new_position, previous_position.copy()

    def run(self):
        cfg = self.cfg

        # Lines 2-4: initial fitness, leader, prey/home positions.
        self._evaluate_population()
        leader = self.best_position.copy()
        prey = leader.copy()
        home = self.rng.uniform(
            self.lower,
            self.upper,
            size=cfg.dimensions,
        )

        tm = 0

        # Line 6: T = 60 * ceil(Dm/10)
        hunting_time = cfg.hunting_time_scale * np.ceil(cfg.dimensions / 10)

        # Lines 7-36: main optimization loop.
        for iteration in range(cfg.max_iterations):
            inertia = self._adaptive_inertia(iteration)

            # Lines 8-10: random ordering/agent preservation.
            order = np.argsort(self.fitness)
            self.positions = self.positions[order]
            self.fitness = self.fitness[order]
            self.details = [self.details[i] for i in order]

            # Line 9: random search agent.
            random_idx = self.rng.integers(0, cfg.population_size)

            old_leader = leader.copy()

            for i in range(cfg.population_size):
                x = self.positions[i].copy()

                # Random variables RN1, RN2, RN3.
                rn1, rn2, rn3 = self.rng.uniform(0.0, 1.0, 3)

                # Select a prey candidate different from current agent.
                prey_idx = self.rng.integers(0, cfg.population_size)
                prey_candidate = self.positions[prey_idx].copy()

                # Searching vs attacking decision.
                if rn2 < rn3:
                    rn4 = self.rng.normal(0.0, 3.0)

                    # SF >= N4 -> searching; otherwise attacking.
                    if rn1 >= rn4:
                        new_x = self._searching_behaviour(
                            x, leader, inertia
                        )
                    else:
                        new_x = self._attacking_behaviour(
                            x, leader, prey_candidate, iteration
                        )
                else:
                    new_x = self._sit_and_wait_behaviour(x, leader)

                # Occasional sit-and-wait / home influence.
                if self.rng.random() < cfg.sit_wait_probability * 0.10:
                    new_x = 0.8 * new_x + 0.2 * home

                self.positions[i] = np.clip(
                    new_x, self.lower, self.upper
                )

            # Lines 25-27: update fitness and leader.
            self._evaluate_population()

            if self.best_fitness < np.inf:
                leader = self.best_position.copy()

            # Lines 28-33: hunting time and leave-and-return behaviour.
            tm += 1
            if tm > self.rng.uniform(0.0, 1.0) * hunting_time:
                # Preserve leader position.
                preserved_leader = leader.copy()

                # Restart a subset of poorer agents.
                n_restart = max(1, cfg.population_size // 10)
                worst = np.argsort(self.fitness)[-n_restart:]

                for idx in worst:
                    self.positions[idx], _ = self._leave_and_return(
                        preserved_leader
                    )

                tm = 0
                self._evaluate_population()
                leader = self.best_position.copy()

            # Lines 34-35: iteration update and global-best leader.
            self.convergence.append(self.best_fitness)

            # Keep an explicit global-best agent in the population.
            worst_idx = int(np.argmax(self.fitness))
            if self.best_position is not None:
                self.positions[worst_idx] = self.best_position.copy()
                self.fitness[worst_idx] = self.best_fitness
                self.details[worst_idx] = self.best_details.copy()

            # Keep variables referenced by the supplied pseudocode.
            _ = random_idx, old_leader

        # Lines 37-40: return global-best solution as optimized CNs.
        optimized_cns = np.sort(
            self.best_details["control_nodes"]
        )

        return {
            "control_nodes": optimized_cns,
            "fitness": self.best_fitness,
            "details": self.best_details,
            "convergence": np.asarray(self.convergence),
            "best_position": self.best_position,
        }
