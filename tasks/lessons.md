# Lessons

- When a user points out that a fix is only partial, audit the full state transition instead of the local write path. For manifold-backed features, verify not only that coordinates are written once, but that persisted artifacts remain valid after the manifold basis changes later.
- When tightening a research-facing claim, audit code enforcement and documentation together. Warning-only guards and stale reference text are both partial fixes if the behavior is meant to be publication-safe.
