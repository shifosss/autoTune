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
                pass

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
                pass

        import torch

        torch.save(state_dict, str(path))
