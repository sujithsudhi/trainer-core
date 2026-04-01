"""Minimal checkpoint helpers for the starter trainer-core package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(path: Path | str, payload: dict[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved)
    return resolved


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return torch.load(resolved, map_location=map_location)


@dataclass
class CheckpointManager:
    directory: Path | str

    def __post_init__(self) -> None:
        self.directory = Path(self.directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_named(self, filename: str, payload: dict[str, Any]) -> Path:
        return save_checkpoint(Path(self.directory) / filename, payload)

    def save_latest(self, payload: dict[str, Any]) -> Path:
        return self.save_named("latest.pt", payload)

    def save_best(self, payload: dict[str, Any]) -> Path:
        return self.save_named("best.pt", payload)

    def resolve_resume_path(self, resume_path: Optional[Path | str] = None) -> Path:
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
