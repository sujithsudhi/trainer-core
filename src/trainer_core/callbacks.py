"""Callback primitives for the reusable training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from .batching import BatchAdapter


@dataclass
class TrainerState:
    """
    Mutable state object shared with callbacks during a training run.
    Args:
        model         : Model being trained or evaluated.
        optimizer     : Optimizer responsible for parameter updates.
        config        : Training configuration object associated with the run.
        device        : Device on which the current run is executing.
        batch_adapter : Adapter used to split and move batches.
        scheduler     : Optional learning-rate scheduler attached to the run.
        scaler        : Optional AMP gradient scaler used during optimization.
        history       : Accumulated epoch records collected during the run.
        epoch         : Current epoch number, starting at zero before training begins.
        stage         : Current lifecycle stage such as `train`, `eval`, or `run_end`.
        batch_index   : One-based batch index within the active stage.
        raw_batch     : Original batch received from the dataloader.
        batch         : Batch after device transfer and any adapter processing.
        inputs        : Model inputs derived from the current batch.
        targets       : Supervision targets derived from the current batch.
        outputs       : Model outputs from the current forward pass.
        loss          : Most recent detached loss tensor.
        metrics       : Most recent metrics payload emitted by the trainer.
        train_metrics : Most recent training metrics dictionary.
        val_metrics   : Most recent validation metrics dictionary.
        record        : Most recent epoch-level history record.
        should_stop   : Whether training should stop after the current epoch.
    Returns:
        None
    """

    model         : nn.Module
    optimizer     : Optimizer
    config        : Any
    device        : torch.device
    batch_adapter : BatchAdapter
    scheduler     : Optional[_LRScheduler | ReduceLROnPlateau] = None
    scaler        : Optional[GradScaler] = None
    history       : list[dict[str, Any]] = field(default_factory=list)
    epoch         : int = 0
    stage         : str = ""
    batch_index   : int = 0
    raw_batch     : Any = None
    batch         : Any = None
    inputs        : Any = None
    targets       : Any = None
    outputs       : Any = None
    loss          : Any = None
    metrics       : Optional[dict[str, Any]] = None
    train_metrics : Optional[dict[str, Any]] = None
    val_metrics   : Optional[dict[str, Any]] = None
    record        : Optional[dict[str, Any]] = None
    should_stop   : bool = False


class Callback:
    """Base callback class with no-op hooks."""

    def on_run_start(self,
                     state : TrainerState,
                    ) -> None:
        """
        Handle the beginning of a trainer run.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return

    def on_epoch_start(self,
                       state : TrainerState,
                      ) -> None:
        """
        Handle the beginning of an epoch.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return

    def on_batch_end(self,
                     state : TrainerState,
                    ) -> None:
        """
        Handle the completion of a training or evaluation batch.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return

    def on_eval_end(self,
                    state : TrainerState,
                   ) -> None:
        """
        Handle the completion of an evaluation pass.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return

    def on_epoch_end(self,
                     state : TrainerState,
                    ) -> None:
        """
        Handle the completion of an epoch.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return

    def on_checkpoint_saved(self,
                            state : TrainerState,
                            path  : Path,
                           ) -> None:
        """
        Handle a best-checkpoint save event.
        Args:
            state : Mutable trainer state for the active run.
            path  : Filesystem path for the saved checkpoint.
        Returns:
            None
        """
        return

    def on_run_end(self,
                   state : TrainerState,
                  ) -> None:
        """
        Handle the end of a trainer run.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        return


class CallableLoggerCallback(Callback):
    """Adapter that preserves the existing epoch-level logger callback contract."""

    def __init__(self,
                 log_fn : Callable[[dict[str, Any]], None],
                ) -> None:
        """
        Wrap a simple logger callable in the callback interface.
        Args:
            log_fn : Callable that receives one epoch record at a time.
        Returns:
            None
        """
        self.log_fn = log_fn

    def on_epoch_end(self,
                     state : TrainerState,
                    ) -> None:
        """
        Forward the completed epoch record to the wrapped logger.
        Args:
            state : Mutable trainer state for the active run.
        Returns:
            None
        """
        if state.record is not None:
            self.log_fn(state.record)


def normalize_callbacks(callbacks : Optional[Iterable[Callback]] = None,
                        logger    : Optional[Callable[[dict[str, Any]], None]] = None,
                       ) -> list[Callback]:
    """
    Normalize callback inputs into a concrete callback list.
    Args:
        callbacks : Optional iterable of callback instances supplied by the caller.
        logger    : Optional legacy logger callable to wrap as a callback.
    Returns:
        List of callbacks ready for trainer dispatch.
    """
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
