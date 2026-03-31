# Lessons

- When a user points out that a fix is only partial, audit the full state transition instead of the local write path. For manifold-backed features, verify not only that coordinates are written once, but that persisted artifacts remain valid after the manifold basis changes later.
