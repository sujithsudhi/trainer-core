"""Batch adapters used by the generic training engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class BatchAdapter:
    """Adapter responsible for batch movement, splitting, and model invocation."""

    def move_to_device(self, batch: Any, device: torch.device, non_blocking: bool) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=non_blocking)
        if isinstance(batch, Mapping):
            return {key: self.move_to_device(value, device, non_blocking) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self.move_to_device(value, device, non_blocking) for value in batch)
        if isinstance(batch, list):
            return [self.move_to_device(value, device, non_blocking) for value in batch]
        return batch

    def split_batch(self, batch: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def forward_model(self, model: nn.Module, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            return model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return model(*inputs)
        return model(inputs)

    def count_batch_items(self, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.size(0)
        if isinstance(batch, Mapping):
            for value in batch.values():
                return self.count_batch_items(value)
            return 0
        if isinstance(batch, (list, tuple)) and batch:
            return self.count_batch_items(batch[0])
        return 0

    def count_tokens(self, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.numel()
        if isinstance(batch, Mapping):
            for value in batch.values():
                return self.count_tokens(value)
            return 0
        if isinstance(batch, (list, tuple)) and batch:
            return self.count_tokens(batch[0])
        return 0


class DefaultBatchAdapter(BatchAdapter):
    """Default adapter supporting tuple/list and common mapping-style batches."""

    def split_batch(self, batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Expected (inputs, targets) from the dataloader.")
            return batch[0], batch[1]
        if isinstance(batch, Mapping):
            if "inputs" in batch and "targets" in batch:
                return batch["inputs"], batch["targets"]
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]
        raise TypeError("Unsupported batch structure; provide (inputs, targets).")


class KeyedBatchAdapter(DefaultBatchAdapter):
    """Mapping adapter for batches keyed with custom input/target field names."""

    def __init__(self, input_key: str, target_key: str) -> None:
        self.input_key = input_key
        self.target_key = target_key

    def split_batch(self, batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, Mapping) and self.input_key in batch and self.target_key in batch:
            return batch[self.input_key], batch[self.target_key]
        return super().split_batch(batch)


__all__ = [
    "BatchAdapter",
    "DefaultBatchAdapter",
    "KeyedBatchAdapter",
]
