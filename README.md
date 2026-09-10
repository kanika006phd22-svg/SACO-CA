# SACO-CA: Self-Adaptive Cheetah Optimization-Based Clustering

This repository contains a Python reference implementation of the **Self-Adaptive Cheetah Optimization-Based Clustering Algorithm (SACO-CA)** based on the supplied manuscript pseudocode.

## What is implemented?

The implementation covers:

1. Initialization of search-agent positions.
2. Fitness evaluation using the proposed Equation (2).
3. Global-best/leader selection.
4. Prey and population-home initialization.
5. Hunting-time calculation.
6. Searching behaviour.
7. Attacking behaviour.
8. Sit-and-wait behaviour.
9. Leave-and-return behaviour.
10. Adaptive inertia.
11. Global-best preservation.
12. Continuous-to-discrete mapping for Control Node (CN) selection.
13. Duplicate removal and feasibility repair.
14. Convergence recording.
15. Export of optimized CNs and fitness results.

## Fitness function

The implemented fitness function is:

F(CH_i) =
w1(1 - E_i/E_max)
+ w2(D_i/D_max)
+ w3(CB_i/CB_max)
+ w4(1 - Q_i/Q_max)

The default manuscript weights are:

- w1 = 0.35: residual energy
- w2 = 0.25: communication distance
- w3 = 0.20: cluster balance
- w4 = 0.20: FSO link quality

Lower fitness is better.

## Discrete Control Node mapping

The optimizer internally uses continuous search-agent positions, while the actual solution must contain valid sensor-node IDs.

The implementation therefore:

- rounds continuous positions to node indices;
- clips indices to the valid range;
- removes duplicate node IDs;
- fills missing positions with unused feasible nodes;
- returns exactly K Control Nodes.

This makes the implementation reproducible for the discrete CN-selection problem.

## Project structure

```text
SACO_CA_GitHub/
├── README.md
├── requirements.txt
├── run_saco_ca.py
├── saco_ca/
│   ├── __init__.py
│   ├── config.py
│   ├── network.py
│   ├── fitness.py
│   └── optimizer.py
└── outputs/
```

## Installation

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SACO_CA_GitHub
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the default experiment

```bash
python run_saco_ca.py
```

Default configuration:

- Sensor nodes: 500
- Control Nodes: 40
- Population size: 30
- Maximum iterations: 300
- Initial energy: 2 J/node
- Random seed: 42

## Change experiment parameters

Example:

```bash
python run_saco_ca.py --nodes 500 --cn 40 --population 50 --iterations 300 --seed 42
```

## Output files

The program creates:

```text
outputs/
├── optimized_control_nodes.csv
├── fitness_summary.csv
├── convergence.csv
└── saco_ca_convergence.png
```

## Important reproducibility note

The supplied manuscript pseudocode references equations/variables such as `SF`, `TR`, `TU`, and `N4`, while the exact mathematical update equations are not visible in the provided material. The repository therefore implements explicit numerical cheetah-inspired update operators for these behaviours rather than claiming a verbatim reconstruction of an unseen equation.

For a journal/patent submission, replace these operators with the exact equations from the final mathematical formulation if those equations exist elsewhere in the manuscript.

## Relationship to NS-3/ONOS

This repository is the Python optimization/reference layer. It can be connected to an NS-3/ONOS experiment through files, sockets, REST APIs, or a Python orchestration layer.

It should not be described as a complete NS-3 + ONOS implementation unless the corresponding simulator/controller integration is added.

## Citation

If this implementation is used in an academic manuscript, cite the associated SACO-CA paper and specify the exact GitHub commit/version used for the experiments.
