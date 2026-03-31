"""Experiment runner: git worktree isolation and checkpoint operations."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages git worktrees for experiment isolation and checkpoint file operations.

    Each experiment runs in its own git worktree checked out at a specific
    commit, ensuring full isolation between concurrent experiments.

    Args:
        project_root: Root directory of the git repository.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._wt_base = project_root / ".autotune_worktrees"
        self._wt_base.mkdir(exist_ok=True)

    def worktree_path(self, node_id: str) -> Path:
        """Return the worktree path for a given node ID.

        Args:
            node_id: Unique experiment node identifier.

        Returns:
            Absolute path to the worktree directory.
        """
        return self._wt_base / node_id

    def create_worktree(self, base_commit: str, node_id: str) -> Path:
        """Create a detached git worktree at the given commit.

        Args:
            base_commit: Git commit SHA to check out.
            node_id: Unique experiment node identifier used as directory name.

        Returns:
            Path to the created worktree directory.

        Raises:
            subprocess.CalledProcessError: If the git worktree add command fails.
        """
        wt_path = self.worktree_path(node_id)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), base_commit],
            cwd=self._root,
            capture_output=True,
            check=True,
        )
        logger.debug("Created worktree %s at commit %s", wt_path, base_commit)
        return wt_path

    def apply_overlay(
        self,
        wt_path: Path,
        overlay_dir: Path,
        deletions: list[str],
    ) -> None:
        """Copy overlay files into the worktree, skipping runtime-generated files.

        Args:
            wt_path: Path to the target worktree.
            overlay_dir: Directory containing files to overlay onto the worktree.
            deletions: Relative file paths within the worktree to delete.
        """
        _SKIP = {"spec.json", "run.log", "result.json"}
        if overlay_dir.exists():
            for src_file in overlay_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel = src_file.relative_to(overlay_dir)
                if rel.name in _SKIP:
                    continue
                dest = wt_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest)
                logger.debug("Overlaid %s -> %s", src_file, dest)

        for deletion in deletions:
            target = wt_path / deletion
            if target.exists():
                target.unlink()
                logger.debug("Deleted overlay target %s", target)

    def collect_result(self, wt_path: Path, archive_dir: Path) -> dict | None:
        """Copy result.json from the worktree to the archive directory and parse it.

        Args:
            wt_path: Path to the worktree containing result.json.
            archive_dir: Destination directory for the archived result.

        Returns:
            Parsed result dict, or None if result.json is absent.
        """
        result_path = wt_path / "result.json"
        if not result_path.exists():
            logger.debug("No result.json found in %s", wt_path)
            return None
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / "result.json"
        shutil.copy2(result_path, dest)
        with open(dest) as f:
            data: dict = json.load(f)
        logger.debug("Collected result from %s: %s", wt_path, data)
        return data

    def collect_checkpoint(
        self,
        checkpoint_src: Path,
        checkpoint_store: Path,
        node_id: str,
    ) -> str | None:
        """Copy a checkpoint directory into the checkpoint store under node_id.

        Args:
            checkpoint_src: Source directory produced by the training script.
            checkpoint_store: Root directory where checkpoints are persisted.
            node_id: Unique experiment node identifier used as subdirectory name.

        Returns:
            The node_id string on success, or None if checkpoint_src does not exist.
        """
        if not checkpoint_src.exists():
            logger.debug("Checkpoint source %s does not exist", checkpoint_src)
            return None
        dest = checkpoint_store / node_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(checkpoint_src, dest)
        logger.debug("Collected checkpoint %s -> %s", checkpoint_src, dest)
        return node_id

    def prepare_branch(
        self,
        checkpoint_id: str,
        checkpoint_store: Path,
        wt_path: Path,
    ) -> None:
        """Stage a parent checkpoint inside the worktree for the training script.

        Copies the stored checkpoint into `.parent_checkpoint/` within the
        worktree so the training script can read it via the env var set by
        the orchestrator.

        Args:
            checkpoint_id: Identifier of the checkpoint to stage.
            checkpoint_store: Root directory where checkpoints are persisted.
            wt_path: Path to the target worktree.
        """
        src = checkpoint_store / checkpoint_id
        if not src.exists():
            logger.debug("Checkpoint %s not found in store %s", checkpoint_id, checkpoint_store)
            return
        dest = wt_path / ".parent_checkpoint"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        logger.debug("Staged parent checkpoint %s -> %s", src, dest)

    def cleanup_worktree(self, wt_path: Path) -> None:
        """Remove a git worktree, falling back to force-remove and prune on failure.

        Args:
            wt_path: Path to the worktree to remove.
        """
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=self._root,
                capture_output=True,
                check=True,
            )
            logger.debug("Removed worktree %s", wt_path)
        except subprocess.CalledProcessError:
            if wt_path.exists():
                shutil.rmtree(wt_path)
            self.prune_stale_worktrees()

    def prune_stale_worktrees(self) -> None:
        """Run git worktree prune to clean up stale worktree metadata."""
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self._root,
            capture_output=True,
        )
        logger.debug("Pruned stale worktrees in %s", self._root)

    def _load_state_dict(self, path: Path) -> dict:
        """Load a model state_dict from a file or directory.

        Supports .pt, .bin, .safetensors files and HuggingFace directories.
        Raises ImportError if neither torch nor safetensors is installed.

        Args:
            path: File path or directory containing the checkpoint.

        Returns:
            The loaded state dict.

        Raises:
            FileNotFoundError: If no checkpoint file found in the directory.
            ImportError: If torch is not installed.
        """
        if path.is_dir():
            candidates = [
                "adapter_model.safetensors",
                "pytorch_model.bin",
                "model.safetensors",
            ]
            for name in candidates:
                if (path / name).exists():
                    return self._load_state_dict(path / name)
            raise FileNotFoundError(f"No checkpoint file found in {path}")

        # Try safetensors first
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                return load_file(str(path))
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: uv pip install -e '.[safetensors]'"
                )

        # Fall back to torch
        try:
            import torch

            return torch.load(str(path), weights_only=True)
        except (ImportError, ModuleNotFoundError):
            pass

        raise ImportError(
            "torch is required for checkpoint operations. "
            "Install with: uv pip install -e '.[torch]'"
        )

    def _save_state_dict(self, state_dict: dict, path: Path, fmt: str) -> None:
        """Save a state_dict to disk.

        Args:
            state_dict: The state dict to save.
            path: Destination file path.
            fmt: Format — "safetensors" or "pytorch".
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "safetensors":
            try:
                from safetensors.torch import save_file

                save_file(state_dict, str(path))
                return
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "safetensors is required to save in safetensors format. "
                    "Install with: uv pip install -e '.[safetensors]'"
                )

        import torch

        torch.save(state_dict, str(path))

    def _load_adapter_config(self, checkpoint_dir: Path) -> dict[str, object]:
        """Load adapter metadata required for arithmetic checkpoint operations."""
        config_path = checkpoint_dir / "adapter_config.json"
        if not config_path.exists():
            raise ValueError(
                f"Checkpoint {checkpoint_dir} is missing adapter_config.json. "
                "Checkpoint arithmetic only supports adapter-style checkpoints."
            )
        with open(config_path) as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Checkpoint {checkpoint_dir} has an invalid adapter_config.json")
        return config

    @staticmethod
    def _tensor_shape(tensor: object) -> tuple[object, ...]:
        shape = getattr(tensor, "shape", None)
        if shape is None:
            return ()
        return tuple(shape)

    @staticmethod
    def _tensor_dtype(tensor: object) -> str:
        dtype = getattr(tensor, "dtype", None)
        if dtype is None:
            return type(tensor).__name__
        return str(dtype)

    @staticmethod
    def _preview_keys(keys: set[str]) -> str:
        preview = ", ".join(sorted(keys)[:3])
        if len(keys) > 3:
            preview += ", ..."
        return preview or "<none>"

    def _validate_adapter_group(
        self,
        checkpoints: list[tuple[str, Path, dict]],
    ) -> dict[str, object]:
        """Validate that arithmetic operands are compatible adapter checkpoints."""
        if not checkpoints:
            raise ValueError("No checkpoints provided for arithmetic operation.")

        metadata_fields = (
            "base_model_name_or_path",
            "peft_type",
            "task_type",
            "target_modules",
        )
        reference_name, reference_dir, reference_state = checkpoints[0]
        reference_config = self._load_adapter_config(reference_dir)
        reference_keys = set(reference_state)

        for name, checkpoint_dir, state in checkpoints[1:]:
            config = self._load_adapter_config(checkpoint_dir)
            for field in metadata_fields:
                if reference_config.get(field) != config.get(field):
                    raise ValueError(
                        f"Incompatible adapter metadata for {field}: "
                        f"{reference_name} != {name}"
                    )

            state_keys = set(state)
            if state_keys != reference_keys:
                missing = reference_keys - state_keys
                extra = state_keys - reference_keys
                raise ValueError(
                    f"Incompatible tensor keys between {reference_name} and {name}. "
                    f"Missing: {self._preview_keys(missing)}; "
                    f"Extra: {self._preview_keys(extra)}"
                )

            for key in sorted(reference_keys):
                ref_tensor = reference_state[key]
                tensor = state[key]
                ref_shape = self._tensor_shape(ref_tensor)
                shape = self._tensor_shape(tensor)
                if ref_shape != shape:
                    raise ValueError(
                        f"Incompatible tensor shape for key {key}: "
                        f"{reference_name}{ref_shape} != {name}{shape}"
                    )
                ref_dtype = self._tensor_dtype(ref_tensor)
                dtype = self._tensor_dtype(tensor)
                if ref_dtype != dtype:
                    raise ValueError(
                        f"Incompatible tensor dtype for key {key}: "
                        f"{reference_name}({ref_dtype}) != {name}({dtype})"
                    )

        return reference_config

    def _write_adapter_checkpoint(
        self,
        state_dict: dict,
        dest_path: Path,
        adapter_config: dict[str, object],
    ) -> None:
        """Persist an arithmetic checkpoint result with its adapter metadata."""
        dest_path.mkdir(parents=True, exist_ok=True)
        self._save_state_dict(state_dict, dest_path / "pytorch_model.bin", "pytorch")
        with open(dest_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2, sort_keys=True)
            f.write("\n")

    def merge_checkpoints(
        self,
        parent_ids: tuple[str, ...],
        weights: tuple[float, ...],
        strategy: str,
        checkpoint_store: Path,
        dest_path: Path,
    ) -> None:
        """Merge multiple parent checkpoints into a single checkpoint.

        Assumes all parent checkpoints share the same model architecture,
        tokenizer, and base initialization. Designed for adapter-style
        (LoRA/PEFT) checkpoints from a shared base model. Results are
        undefined for divergent architectures.

        Args:
            parent_ids: IDs of parent nodes whose checkpoints to merge.
            weights: Per-parent weights for merging.
            strategy: One of "weighted_avg", "compose", "selective".
            checkpoint_store: Root directory containing per-node checkpoint dirs.
            dest_path: Where to write the merged checkpoint.

        Raises:
            ImportError: If torch is not installed.
            NotImplementedError: If strategy is "selective".
            ValueError: If strategy is unknown or checkpoints are incompatible.
        """
        parent_dirs = [checkpoint_store / pid for pid in parent_ids]
        state_dicts = [self._load_state_dict(path) for path in parent_dirs]
        adapter_config = self._validate_adapter_group(
            list(zip(parent_ids, parent_dirs, state_dicts, strict=True))
        )

        if strategy == "weighted_avg":
            merged = {}
            for key in state_dicts[0]:
                merged[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
            self._write_adapter_checkpoint(merged, dest_path, adapter_config)

        elif strategy == "compose":
            base = state_dicts[0]
            accumulated_delta = {k: base[k] * 0.0 for k in base}
            for sd in state_dicts[1:]:
                for k in accumulated_delta:
                    accumulated_delta[k] = accumulated_delta[k] + (sd[k] - base[k])
            composed = {k: base[k] + accumulated_delta[k] for k in base}
            self._write_adapter_checkpoint(composed, dest_path, adapter_config)

        elif strategy == "selective":
            raise NotImplementedError(
                "selective merge requires layer_manifest config (not yet supported)"
            )
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        logger.info("Merged checkpoints %s via %s -> %s", parent_ids, strategy, dest_path)

    def graft_checkpoint(
        self,
        source_id: str,
        source_parent_id: str,
        target_parent_id: str,
        checkpoint_store: Path,
        dest_path: Path,
    ) -> None:
        """Apply a technique delta from one branch to another parent.

        Computes delta = source - source_parent, then grafted = target_parent + delta.

        Assumes all three checkpoints share the same model architecture and
        base initialization. Designed for adapter-style (LoRA/PEFT) checkpoints.
        Results are undefined for divergent architectures.

        Args:
            source_id: ID of the experiment whose technique delta to apply.
            source_parent_id: ID of the source's parent checkpoint.
            target_parent_id: ID of the target parent checkpoint to apply the delta onto.
            checkpoint_store: Root directory containing per-node checkpoint dirs.
            dest_path: Where to write the grafted checkpoint.

        Raises:
            ImportError: If torch is not installed.
            ValueError: If checkpoints have incompatible key sets.
        """
        checkpoint_dirs = {
            "source": checkpoint_store / source_id,
            "source_parent": checkpoint_store / source_parent_id,
            "target_parent": checkpoint_store / target_parent_id,
        }
        source = self._load_state_dict(checkpoint_store / source_id)
        source_parent = self._load_state_dict(checkpoint_store / source_parent_id)
        target_parent = self._load_state_dict(checkpoint_store / target_parent_id)
        adapter_config = self._validate_adapter_group(
            [
                (source_id, checkpoint_dirs["source"], source),
                (source_parent_id, checkpoint_dirs["source_parent"], source_parent),
                (target_parent_id, checkpoint_dirs["target_parent"], target_parent),
            ]
        )

        grafted = {}
        for key in target_parent:
            delta = source[key] - source_parent[key]
            grafted[key] = target_parent[key] + delta

        self._write_adapter_checkpoint(grafted, dest_path, adapter_config)
        logger.info(
            "Grafted %s (delta from %s) onto %s -> %s",
            source_id,
            source_parent_id,
            target_parent_id,
            dest_path,
        )
