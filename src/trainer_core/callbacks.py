"""Callback primitives for the local training engine refactor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch import nn

from .batching import BatchAdapter


@dataclass
class TrainerState:
    """Mutable state object shared with callbacks during a training run."""

    model: nn.Module
    optimizer: Optimizer
    config: Any
    device: torch.device
    batch_adapter: BatchAdapter
    scheduler: Optional[_LRScheduler | ReduceLROnPlateau] = None
    scaler: Optional[GradScaler] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    epoch: int = 0
    stage: str = ""
    batch_index: int = 0
    raw_batch: Any = None
    batch: Any = None
    inputs: Any = None
    targets: Any = None
    outputs: Any = None
    loss: Any = None
    metrics: Optional[dict[str, Any]] = None
    train_metrics: Optional[dict[str, Any]] = None
    val_metrics: Optional[dict[str, Any]] = None
    record: Optional[dict[str, Any]] = None
    should_stop: bool = False


class Callback:
    """Base callback class with no-op hooks."""

    def on_run_start(self, state: TrainerState) -> None:
        return

    def on_epoch_start(self, state: TrainerState) -> None:
        return

    def on_batch_end(self, state: TrainerState) -> None:
        return

    def on_eval_end(self, state: TrainerState) -> None:
        return

    def on_epoch_end(self, state: TrainerState) -> None:
        return

    def on_checkpoint_saved(self, state: TrainerState, path: Path) -> None:
        return

    def on_run_end(self, state: TrainerState) -> None:
        return


class CallableLoggerCallback(Callback):
    """Adapter that preserves the existing epoch-level logger callback contract."""

    def __init__(self, log_fn: Callable[[dict[str, Any]], None]) -> None:
        self.log_fn = log_fn

    def on_epoch_end(self, state: TrainerState) -> None:
        if state.record is not None:
            self.log_fn(state.record)


def normalize_callbacks(
    callbacks: Optional[Iterable[Callback]] = None,
    logger: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[Callback]:
    resolved = list(callbacks or [])
    if logger is not None:
        resolved.append(CallableLoggerCallback(logger))
    return resolved


__all__ = [
    "Callback",
    "CallableLoggerCallback",
    "TrainerState",
    "normalize_callbacks",
]
