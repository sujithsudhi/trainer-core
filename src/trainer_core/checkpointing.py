"""Minimal checkpoint helpers for the starter trainer-core package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(path    : Path | str,
                    payload : dict[str, Any],
                   ) -> Path:
    """
    Save a checkpoint payload to disk.
    Args:
        path    : Filesystem location for the serialized checkpoint.
        payload : Checkpoint data to persist with `torch.save`.
    Returns:
        Resolved checkpoint path on disk.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved)
    return resolved


def load_checkpoint(path         : Path | str,
                    map_location : str | torch.device = "cpu",
                   ) -> dict[str, Any]:
    """
    Load a checkpoint payload from disk.
    Args:
        path         : Filesystem location for the serialized checkpoint.
        map_location : Device mapping passed through to `torch.load`.
    Returns:
        Deserialized checkpoint payload.
    Raises:
        FileNotFoundError: Raised when the requested checkpoint path does not exist.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return torch.load(resolved, map_location=map_location)


@dataclass
class CheckpointManager:
    """
    Helper for saving and resolving trainer checkpoints inside one directory.
    Args:
        directory : Base directory used for saving and resolving checkpoints.
    Returns:
        None
    """

    directory : Path | str

    def __post_init__(self) -> None:
        """
        Resolve and create the checkpoint directory.
        Args:
            None
        Returns:
            None
        """
        self.directory = Path(self.directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_named(self,
                   filename : str,
                   payload  : dict[str, Any],
                  ) -> Path:
        """
        Save a named checkpoint file inside the managed directory.
        Args:
            filename : Checkpoint filename relative to the managed directory.
            payload  : Checkpoint data to persist with `torch.save`.
        Returns:
            Resolved checkpoint path on disk.
        """
        return save_checkpoint(Path(self.directory) / filename, payload)

    def save_latest(self,
                    payload : dict[str, Any],
                   ) -> Path:
        """
        Save the conventional latest checkpoint file.
        Args:
            payload : Checkpoint data to persist with `torch.save`.
        Returns:
            Resolved checkpoint path on disk.
        """
        return self.save_named("latest.pt", payload)

    def save_best(self,
                  payload : dict[str, Any],
                 ) -> Path:
        """
        Save the conventional best checkpoint file.
        Args:
            payload : Checkpoint data to persist with `torch.save`.
        Returns:
            Resolved checkpoint path on disk.
        """
        return self.save_named("best.pt", payload)

    def resolve_resume_path(self,
                            resume_path : Optional[Path | str] = None,
                           ) -> Path:
        """
        Resolve a resume path relative to the managed checkpoint directory.
        Args:
            resume_path : Optional explicit resume path or filename.
        Returns:
            Resolved checkpoint path for resuming training.
        """
        if resume_path is None:
            return Path(self.directory) / "latest.pt"
        resolved = Path(resume_path)
        if resolved.is_absolute():
            return resolved
        return Path(self.directory) / resolved


__all__ = [
    "CheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
]
