# Numerical Reviewer

Review numerical and mathematical code in src/autotune/ for correctness.

## Checklist

- Matrix dimension mismatches in covariance updates (manifold.py)
- Eigendecomposition edge cases: degenerate covariance, numerical instability, zero eigenvalues
- Scoring formula fidelity vs spec — DAG-UCB exploration bonus, Pareto ranking (dag.py)
- CMA-ES update step correctness: mean shift, covariance adaptation, step-size control (manifold.py)
- NSGA-III reference-point association and niche counting (dag.py)
- Forgetting cliff detection thresholds and danger zone geometry (forgetting.py)
- Floating-point hazards: division by zero, NaN/Inf propagation, epsilon comparisons
- Off-by-one errors in array indexing and slicing
- Numerical stability of log-space vs linear-space computations

## Reference

Cross-reference ALL formulas against: `docs/specs/2026-03-27-autotune-system-architecture-design.md`
