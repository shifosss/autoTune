# Integration Validator

Validate data flow across autoTune module boundaries. The module dependency graph is:

```
cli.py -> dag.py, navigator.py, orchestrator.py, viz/server.py
navigator.py -> dag.py, manifold.py, forgetting.py
orchestrator.py -> runner.py, manifold.py, forgetting.py
dag.py -> standalone
manifold.py -> standalone
forgetting.py -> standalone
runner.py -> standalone
```

## Checklist

- DAGNode and other frozen dataclasses from types.py are never mutated at any call site
- Navigator correctly composes: DAG-UCB scores + manifold theta proposals + forgetting danger zones
- Orchestrator sets env vars correctly: AUTOTUNE_PARENT_CHECKPOINT, AUTOTUNE_CHECKPOINT_DIR
- Runner git worktree/branch operations do not corrupt DAG state or leave orphan branches
- CLI subcommands wire to correct module entry points with correct argument forwarding
- Merge operations: multi-parent DAG nodes receive correct lineage from all parents
- Checkpoint lifecycle: create -> train -> evaluate -> score -> rank flows end-to-end
- Config schema (YAML) validated at load time, not silently defaulted
- SSE viz server reads DAG state without holding locks that block experiment submission
