# Todo

- [x] Review current forgetting/manifold/orchestrator code path and confirm the failure mode.
- [x] Create task branch `fix_forgetting_grounding`.
- [x] Implement a single-coordinate-system grounding flow for executed nodes and forgetting zones.
- [x] Add regression tests for early-stage cliff grounding and suggestion filtering.
- [x] Run focused verification (`pytest`, `ruff`) on the touched modules.

## Notes

- Root cause: `_post_completion_update()` only writes `manifold_coords` inside the trained-manifold branch (`len(executed) >= 10`), but `forgetting.check_cliff()` runs for completed nodes regardless of manifold generation.
- Naively projecting only before training is incomplete because danger zones created in generation-0 coordinates become stale once the manifold later updates its mean/basis.
- Target fix: synchronize all executed node coordinates against the current manifold state every tick, then rebuild the forgetting detector from DAG history so all zones live in the same coordinate system.

## Review

- Tightened forgetting rebuild semantics in [orchestrator.py](/Users/chenzhang/Alex's%20Codebases/autoTune/src/autotune/orchestrator.py) so danger zones are rebuilt from executed DAG history whenever a completion lands in a configured parameter space, even if that specific completion omits `retention_score`.
- Preserved a single active coordinate system: executed-node `manifold_coords` are synced from the current manifold state, and danger zones are regenerated against those coordinates instead of relying on stale incremental appends.
- Extended [test_orchestrator.py](/Users/chenzhang/Alex's%20Codebases/autoTune/tests/test_orchestrator.py) with coverage for:
- early-stage cliffs getting grounded non-empty danger-zone centers
- early-stage danger zones filtering later merge suggestions
- manifold retraining rebuilding an earlier cliff even when the 10th completion lacks a retention metric
- Verification:
- `env UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/autotune/orchestrator.py tests/test_orchestrator.py`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py tests/test_navigator.py tests/test_forgetting.py tests/test_manifold.py -q`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
