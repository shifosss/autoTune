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

- Tightened forgetting rebuild semantics in [orchestrator.py](src/autotune/orchestrator.py) so danger zones are rebuilt from executed DAG history whenever a completion lands in a configured parameter space, even if that specific completion omits `retention_score`.
- Preserved a single active coordinate system: executed-node `manifold_coords` are synced from the current manifold state, and danger zones are regenerated against those coordinates instead of relying on stale incremental appends.
- Extended [test_orchestrator.py](tests/test_orchestrator.py) with coverage for:
- early-stage cliffs getting grounded non-empty danger-zone centers
- early-stage danger zones filtering later merge suggestions
- manifold retraining rebuilding an earlier cliff even when the 10th completion lacks a retention metric
- Verification:
- `env UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/autotune/orchestrator.py tests/test_orchestrator.py`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator.py tests/test_navigator.py tests/test_forgetting.py tests/test_manifold.py -q`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`

---

## Problem 6: Checkpoint Compatibility Hardening

- [x] Review current runner checkpoint arithmetic, runner tests, and workflow docs.
- [x] Create task branch `fix_checkpoint_compatibility`.
- [x] Add strict adapter compatibility validation in `src/autotune/runner.py`.
- [x] Make `weighted_avg`, `compose`, and `graft` fail closed on metadata/key/shape/dtype mismatch.
- [x] Expand `tests/test_runner.py` for valid adapter ops and hard failures.
- [x] Narrow checkpoint arithmetic claims in `docs/workflow-reference.md` and `docs/workflow-flowchart.html`.
- [x] Run focused and full verification.
- [x] Document review results here after verification.

## Notes

- Current `weighted_avg` only warns on low key overlap and still proceeds.
- Current `compose` assumes base-relative delta composition but does not validate adapter metadata or exact tensor compatibility.
- Current `graft` validates overlap only between source and source parent; target compatibility is still permissive.
- Docs still present checkpoint arithmetic too broadly for publication-safe claims.

## Review

- `src/autotune/runner.py` now uses a shared adapter compatibility validator for
  all arithmetic checkpoint operations.
- `weighted_avg`, `compose`, and `graft` now fail closed unless every operand:
  has `adapter_config.json`, matches adapter metadata, and exposes identical
  tensor keys, shapes, and dtypes.
- Arithmetic outputs now persist `adapter_config.json` alongside the generated
  `pytorch_model.bin`, so merged/grafted parents remain valid adapter
  checkpoints for downstream training.
- `tests/test_runner.py` now covers:
  - successful adapter-compatible `weighted_avg`, `compose`, and `graft`
  - missing `adapter_config.json`
  - mismatched `base_model_name_or_path`
  - mismatched `peft_type`
  - key-set mismatch for `weighted_avg` and `compose`
  - shape mismatch
  - dtype mismatch
  - graft target missing the required delta key
- `docs/workflow-reference.md` and `docs/workflow-flowchart.html` now document
  checkpoint arithmetic as adapter-only with strict compatibility
  preconditions, and they no longer describe dropping or silently preserving
  incompatible keys.
- Verification:
  - `env UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/autotune/runner.py tests/test_runner.py`
  - `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runner.py -q`
  - `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
