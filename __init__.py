"""
SACO-CA: Self-Adaptive Cheetah Optimization-Based Clustering.

Reference implementation based on the supplied SACO-CA pseudocode:
- fitness evaluation using residual energy, distance, cluster balance,
  and FSO link quality;
- searching, attacking, sit-and-wait, and leave-and-return behaviours;
- global-best preservation;
- discrete control-node (CN) repair for sensor-node indices.
"""

__version__ = "1.0.0"
