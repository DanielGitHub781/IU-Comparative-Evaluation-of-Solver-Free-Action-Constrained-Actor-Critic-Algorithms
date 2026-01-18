# IU-Comparative-Evaluation-of-Solver-Free-Action-Constrained-Actor-Critic-Algorithms

This repository contains the experimental code used for the bachelor thesis  
**“A Comparative Evaluation of Solver-Free Action-Constrained Actor–Critic Algorithms.”**

## Code Origin and Acknowledgment

The majority of the code in this repository is based on the official implementation of the benchmark study:

**Kasaura et al. (2023)** – *Benchmarking Actor-Critic Deep Reinforcement Learning Algorithms for Robotics Control With Action Constraints*  
GitHub repository:  
https://github.com/omron-sinicx/action-constrained-RL-benchmark

The original benchmark code provides the core implementations of solver-free action-constrained actor–critic algorithms, constraint handling mechanisms (such as α-projection and radial squashing), environment wrappers and training pipelines. All credit for the original algorithm implementations and benchmark design belongs to the authors of that work.

## Modifications and Extensions

The code was adapted and extended for the purposes of this bachelor thesis. The main modifications include:

- Removal of the external Chebyshev center solver dependency to ensure full reproducibility without proprietary software.
- Implementation of a deterministic fallback mechanism for constraint center handling under symmetric constraint sets.
- Unified metrics aggregation and visualization scripts to support multi-seed evaluation and consistent result reporting.
- Extensions to support the specific evaluation metrics and analyses presented in the thesis.
```
