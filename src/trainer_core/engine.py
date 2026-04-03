"""General training utilities built on PyTorch."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import json
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from .batching import BatchAdapter, DefaultBatchAdapter
from .callbacks import Callback, TrainerState, normalize_callbacks

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore


_DEFAULT_BATCH_ADAPTER = DefaultBatchAdapter()


_TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs"                       : 5,
    "device"                       : "auto",
    "gradient_clip_norm"           : None,
    "gradient_accumulation_steps"  : 1,
    "use_amp"                      : "auto",
    "amp_dtype"                    : "auto",
    "log_interval"                 : 50,
    "non_blocking"                 : True,
    "early_stopping_patience"      : 10,
    "lr_reduction_patience"        : 5,
    "lr_reduction_factor"          : 0.5,
    "warmup_epochs"                : 0,
    "warmup_start_factor"          : 0.1,
    "use_cosine_decay"             : False,
    "min_lr"                       : 0.0,
}


''' Function: _resolve_bool
    Description: Convert various input types to boolean value with auto-detection support.
    Args:
        value   : Input value to convert.
        default : Default boolean value if auto is specified.
    Returns:
        Resolved boolean value.
'''
def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return default
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


''' Function: _resolve_device
    Description: Resolve device string with automatic CUDA detection.
    Args:
        value : Device specification string or object.
    Returns:
        Resolved device string ('cuda' or 'cpu').
'''
def _resolve_device(value: Any) -> str:
    if value is None:
        value = "auto"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value.strip()
    return str(value)


''' Function: _resolve_positive_int
    Description: Convert value to positive integer with minimum bound of 1.
    Args:
        value   : Input value to convert.
        default : Default value if input is None.
    Returns:
        Positive integer value.
'''
def _resolve_positive_int(value: Any, default: int) -> int:
    candidate = default if value is None else int(value)
    return max(1, candidate)


def _resolve_non_negative_int(value: Any, default: int) -> int:
    candidate = default if value is None else int(value)
    return max(0, candidate)


def _resolve_early_stopping_patience(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "off", "false"}:
            return None
        value = lowered
    try:
        patience = int(value)
    except (TypeError, ValueError):
        patience = int(float(value))
    return patience if patience > 0 else None


def _resolve_lr_reduction_patience(value: Any) -> int | None:
    return _resolve_early_stopping_patience(value)


def _resolve_lr_reduction_factor(value: Any) -> float:
    factor = 0.5 if value is None else float(value)
    if factor <= 0 or factor >= 1:
        raise ValueError("lr_reduction_factor must be between 0 and 1 (exclusive).")
    return factor


def _resolve_non_negative_float(value: Any, default: float) -> float:
    candidate = default if value is None else float(value)
    return max(0.0, candidate)


def _resolve_gradient_clip_norm(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "off", "false"}:
            return None
        clip = float(value)
    else:
        clip = float(value)
    return clip if clip > 0 else None


def _resolve_use_amp(value: Any) -> bool:
    default = torch.cuda.is_available()
    resolved = _resolve_bool(value, default)
    return resolved and torch.cuda.is_available()


def _resolve_amp_dtype(value: Any) -> str:
    if value is None:
        value = _TRAINING_DEFAULTS.get("amp_dtype", "auto")
    if isinstance(value, torch.dtype):
        if value == torch.float16:
            return "float16"
        if value == torch.bfloat16:
            return "bfloat16"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "auto", "fp16", "float16", "half"}:
            return "float16"
        if lowered in {"bf16", "bfloat16"}:
            return "bfloat16"
    raise ValueError(
        "amp_dtype must be one of auto, fp16, float16, bf16, bfloat16, "
        "torch.float16, or torch.bfloat16."
    )


def _default_epochs() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("epochs"), 5)


def _default_device() -> str:
    return _resolve_device(_TRAINING_DEFAULTS.get("device"))


def _default_gradient_clip_norm() -> float | None:
    return _resolve_gradient_clip_norm(_TRAINING_DEFAULTS.get("gradient_clip_norm"))


def _default_gradient_accumulation_steps() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("gradient_accumulation_steps"), 1)


def _default_use_amp() -> bool:
    return _resolve_use_amp(_TRAINING_DEFAULTS.get("use_amp"))


def _default_amp_dtype() -> str:
    return _resolve_amp_dtype(_TRAINING_DEFAULTS.get("amp_dtype"))


def _default_log_interval() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("log_interval"), 50)


def _default_non_blocking() -> bool:
    return _resolve_bool(_TRAINING_DEFAULTS.get("non_blocking"), True)


def _default_early_stopping_patience() -> int | None:
    return _resolve_early_stopping_patience(_TRAINING_DEFAULTS.get("early_stopping_patience"))


def _default_lr_reduction_patience() -> int | None:
    return _resolve_lr_reduction_patience(_TRAINING_DEFAULTS.get("lr_reduction_patience"))


def _default_lr_reduction_factor() -> float:
    return _resolve_lr_reduction_factor(_TRAINING_DEFAULTS.get("lr_reduction_factor"))


def _default_warmup_epochs() -> int:
    return _resolve_non_negative_int(_TRAINING_DEFAULTS.get("warmup_epochs"), 0)


def _default_warmup_start_factor() -> float:
    return _resolve_non_negative_float(_TRAINING_DEFAULTS.get("warmup_start_factor"), 0.1)


def _default_use_cosine_decay() -> bool:
    return _resolve_bool(_TRAINING_DEFAULTS.get("use_cosine_decay"), False)


def _default_min_lr() -> float:
    return _resolve_non_negative_float(_TRAINING_DEFAULTS.get("min_lr"), 0.0)


@dataclass
class TrainingConfig:
    epochs                       : int                = field(default_factory=_default_epochs)
    device                       : str                = field(default_factory=_default_device)
    gradient_clip_norm           : float | None       = field(default_factory=_default_gradient_clip_norm)
    gradient_accumulation_steps  : int                = field(default_factory=_default_gradient_accumulation_steps)
    use_amp                      : bool | str | None  = field(default_factory=_default_use_amp)  # type: ignore[assignment]
    amp_dtype                    : str | torch.dtype | None = field(default_factory=_default_amp_dtype)
    log_interval                 : int                = field(default_factory=_default_log_interval)
    non_blocking                 : bool               = field(default_factory=_default_non_blocking)
    early_stopping_patience      : int | None         = field(default_factory=_default_early_stopping_patience)
    lr_reduction_patience        : int | None         = field(default_factory=_default_lr_reduction_patience)
    lr_reduction_factor          : float              = field(default_factory=_default_lr_reduction_factor)
    warmup_epochs                : int                = field(default_factory=_default_warmup_epochs)
    warmup_start_factor          : float              = field(default_factory=_default_warmup_start_factor)
    use_cosine_decay             : bool               = field(default_factory=_default_use_cosine_decay)
    min_lr                       : float              = field(default_factory=_default_min_lr)

    def __post_init__(self) -> None:
        self.epochs = _resolve_positive_int(self.epochs, 5)
        self.device = _resolve_device(self.device)
        self.gradient_clip_norm = _resolve_gradient_clip_norm(self.gradient_clip_norm)
        self.gradient_accumulation_steps = _resolve_positive_int(
            self.gradient_accumulation_steps, 1
        )
        self.use_amp = _resolve_use_amp(self.use_amp)
        self.amp_dtype = _resolve_amp_dtype(self.amp_dtype)
        self.log_interval = _resolve_positive_int(self.log_interval, 50)
        self.non_blocking = _resolve_bool(self.non_blocking, True)
        self.early_stopping_patience = _resolve_early_stopping_patience(
            self.early_stopping_patience
        )
        self.lr_reduction_patience = _resolve_lr_reduction_patience(
            self.lr_reduction_patience
        )
        self.lr_reduction_factor = _resolve_lr_reduction_factor(self.lr_reduction_factor)
        self.warmup_epochs = _resolve_non_negative_int(self.warmup_epochs, 0)
        self.warmup_start_factor = _resolve_non_negative_float(self.warmup_start_factor, 0.1)
        self.use_cosine_decay = _resolve_bool(self.use_cosine_decay, False)
        self.min_lr = _resolve_non_negative_float(self.min_lr, 0.0)


def load_training_config(
    source: Path | str | Mapping[str, Any] | TrainingConfig | None = None,
) -> TrainingConfig:
    if isinstance(source, TrainingConfig):
        return source
    if isinstance(source, Mapping):
        payload = dict(_TRAINING_DEFAULTS)
        payload.update(source)
    elif source is None:
        payload = dict(_TRAINING_DEFAULTS)
    else:
        resolved = Path(source).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = dict(_TRAINING_DEFAULTS)
            payload.update(json.load(handle))
    valid_keys = {
        "epochs",
        "device",
        "gradient_clip_norm",
        "gradient_accumulation_steps",
        "use_amp",
        "amp_dtype",
        "log_interval",
        "non_blocking",
        "early_stopping_patience",
        "lr_reduction_patience",
        "lr_reduction_factor",
        "warmup_epochs",
        "warmup_start_factor",
        "use_cosine_decay",
        "min_lr",
    }
    kwargs = {key: payload.get(key) for key in valid_keys}
    return TrainingConfig(**kwargs)


def _is_amp_enabled(config: TrainingConfig, device: torch.device) -> bool:
    return bool(config.use_amp) and device.type == "cuda" and torch.cuda.is_available()


def _autocast_dtype(config: TrainingConfig) -> torch.dtype:
    if config.amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def _is_grad_scaling_enabled(config: TrainingConfig, device: torch.device) -> bool:
    return _is_amp_enabled(config, device) and _autocast_dtype(config) == torch.float16


def _create_grad_scaler(config: TrainingConfig, device: torch.device) -> GradScaler:
    return GradScaler(enabled=_is_grad_scaling_enabled(config, device))


def _notify_callbacks(
    callbacks: Iterable[Callback],
    hook_name: str,
    state: Optional[TrainerState],
    *args: Any,
) -> None:
    if state is None:
        return
    for callback in callbacks:
        getattr(callback, hook_name)(state, *args)


''' Function: _move_to_device
    Description: Recursively move batch data to specified device.
    Args:
        batch        : Batch data (tensor, dict, list, or tuple).
        device       : Target device for data.
        non_blocking : Whether to use non-blocking transfer.
    Returns:
        Batch data on target device.
'''
def _move_to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    return _DEFAULT_BATCH_ADAPTER.move_to_device(batch, device, non_blocking)


def _split_batch(batch: Any) -> tuple[Any, Any]:
    return _DEFAULT_BATCH_ADAPTER.split_batch(batch)


def _count_batch_items(batch: Any) -> int:
    return _DEFAULT_BATCH_ADAPTER.count_batch_items(batch)


def _count_tokens(batch: Any) -> int:
    return _DEFAULT_BATCH_ADAPTER.count_tokens(batch)


def _metric_value(metrics: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if metrics is None:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_checkpoint_metric(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    numeric = float(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _best_checkpoint_filename(epoch: int, train_loss: Optional[float], val_loss: Optional[float]) -> str:
    train_tag = _format_checkpoint_metric(train_loss)
    val_tag = _format_checkpoint_metric(val_loss)
    return f"epoch-{epoch:03d}_train-{train_tag}_val-{val_tag}.pt"


def _try_len(iterable: Iterable[Any]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _progress_iter(iterable: Iterable[Any], desc: str) -> tuple[Iterable[Any], Optional[Any]]:
    if tqdm is None:
        return iterable, None
    total = _try_len(iterable)
    bar = tqdm(iterable, desc=desc, total=total, leave=True)
    return bar, bar


def _forward_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    return _DEFAULT_BATCH_ADAPTER.forward_model(model, inputs)


def train_one_epoch(model         : nn.Module,
                    dataloader    : Iterable[Any],
                    optimizer     : Optimizer,
                    loss_fn       : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    config        : TrainingConfig,
                    scaler        : Optional[GradScaler] = None,
                    progress_desc : Optional[str] = None,
                    batch_adapter : Optional[BatchAdapter] = None,
                    callbacks     : Optional[Iterable[Callback]] = None,
                    state         : Optional[TrainerState] = None,
                   ) -> Dict[str, float]:
    
    device = torch.device(config.device)
    batch_adapter = batch_adapter or _DEFAULT_BATCH_ADAPTER
    resolved_callbacks = list(callbacks or [])
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = scaler or _create_grad_scaler(config, device)
    amp_enabled = _is_amp_enabled(config, device)

    accum_steps = max(config.gradient_accumulation_steps, 1)
    total_loss = torch.zeros((), device=device)
    total_examples = 0
    total_tokens = 0
    total_steps = 0
    start_time = time.perf_counter()
    last_logged_loss: Optional[float] = None

    iterator, progress_bar = _progress_iter(dataloader, progress_desc or "Train")
    try:
        for step, raw_batch in enumerate(iterator, start=1):
            batch = batch_adapter.move_to_device(raw_batch, device, config.non_blocking)
            inputs, targets = batch_adapter.split_batch(batch)
            autocast_kwargs: Dict[str, Any] = {
                "device_type": device.type,
                "enabled": amp_enabled,
            }
            if amp_enabled:
                autocast_kwargs["dtype"] = _autocast_dtype(config)
            with autocast(**autocast_kwargs):
                outputs = batch_adapter.forward_model(model, inputs)
                loss = loss_fn(outputs, targets)
                loss_to_backward = loss / accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = step % accum_steps == 0
            if should_step:
                if config.gradient_clip_norm is not None:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_loss = total_loss + loss.detach()
            total_examples += batch_adapter.count_batch_items(raw_batch)
            total_tokens += batch_adapter.count_tokens(inputs)
            total_steps += 1
            if state is not None:
                state.stage = "train"
                state.batch_index = step
                state.raw_batch = raw_batch
                state.batch = batch
                state.inputs = inputs
                state.targets = targets
                state.outputs = outputs
                state.loss = loss.detach()
            _notify_callbacks(resolved_callbacks, "on_batch_end", state)
            if progress_bar is not None:
                should_log = step == 1 or step % config.log_interval == 0
                if should_log:
                    elapsed = max(time.perf_counter() - start_time, 1e-8)
                    last_logged_loss = float(loss.detach().item())
                    progress_bar.set_postfix(
                        {"loss": f"{last_logged_loss:.4f}",
                         "tok/s": f"{int(total_tokens / elapsed):,}"},
                        refresh=False,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if total_steps % accum_steps != 0:
        if config.gradient_clip_norm is not None:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    elapsed = max(time.perf_counter() - start_time, 1e-8)
    total_loss_value = float(total_loss.item())
    metrics = {
        "loss"     : total_loss_value / max(total_steps, 1),
        "loss_sum" : total_loss_value,
        "batches"  : total_steps,
        "examples" : total_examples,
        "tokens"   : total_tokens,
        "tok_per_s": total_tokens / elapsed,
    }
    if state is not None:
        state.metrics = metrics
        state.train_metrics = metrics
    return metrics


''' Function: evaluate
    Description: Evaluate model on validation/test data without gradient computation.
    Args:
        model         : Neural network model to evaluate.
        dataloader    : Evaluation data iterator.
        loss_fn       : Loss function for computing evaluation loss.
        device        : Device for model and data.
        non_blocking  : Whether to use non-blocking data transfer.
        progress_desc : Description for progress bar display.
    Returns:
        Dictionary with evaluation metrics (loss, examples, batches).
'''
def evaluate(model         : nn.Module,
             dataloader    : Iterable[Any],
             loss_fn       : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             device        : str | torch.device,
             non_blocking  : bool = True,
             progress_desc : Optional[str] = None,
             batch_adapter : Optional[BatchAdapter] = None,
             callbacks     : Optional[Iterable[Callback]] = None,
             state         : Optional[TrainerState] = None,
            ) -> Dict[str, float]:
    device = torch.device(device)
    batch_adapter = batch_adapter or _DEFAULT_BATCH_ADAPTER
    resolved_callbacks = list(callbacks or [])
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    iterator, progress_bar = _progress_iter(dataloader, progress_desc or "Eval")
    with torch.no_grad():
        try:
            for step, raw_batch in enumerate(iterator, start=1):
                batch = batch_adapter.move_to_device(raw_batch, device, non_blocking)
                inputs, targets = batch_adapter.split_batch(batch)
                outputs = batch_adapter.forward_model(model, inputs)
                loss = loss_fn(outputs, targets)

                total_loss += loss.item()
                total_examples += batch_adapter.count_batch_items(raw_batch)
                total_steps += 1
                if state is not None:
                    state.stage = "eval"
                    state.batch_index = step
                    state.raw_batch = raw_batch
                    state.batch = batch
                    state.inputs = inputs
                    state.targets = targets
                    state.outputs = outputs
                    state.loss = loss.detach()
                _notify_callbacks(resolved_callbacks, "on_batch_end", state)
                if progress_bar is not None:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    metrics = {
        "loss"     : total_loss / max(total_steps, 1),
        "loss_sum" : total_loss,
        "batches"  : total_steps,
        "examples" : total_examples,
    }
    if state is not None:
        state.metrics = metrics
        state.val_metrics = metrics
    _notify_callbacks(resolved_callbacks, "on_eval_end", state)
    return metrics


class Trainer:
    ''' Function: __init__
        Description: Initialize trainer with model, optimizer, and training components.
        Args:
            model        : Neural network model to train.
            optimizer    : Optimizer for parameter updates.
            loss_fn      : Loss function for training.
            train_loader : Training data loader.
            config       : Training configuration.
            val_loader   : Optional validation data loader.
            scheduler    : Optional learning rate scheduler.
        Returns:
            None
    '''
    def __init__(self,
                 model        : nn.Module,
                 optimizer    : Optimizer,
                 loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 train_loader : Iterable[Any],
                 config       : TrainingConfig,
                 val_loader   : Optional[Iterable[Any]] = None,
                 scheduler    : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
                 logger       : Optional[Callable[[Dict[str, Any]], None]] = None,
                 best_checkpoint_dir: Optional[Path | str] = None,
                 batch_adapter: Optional[BatchAdapter] = None,
                 callbacks    : Optional[Iterable[Callback]] = None,
                ) -> None:
        self.device       = torch.device(config.device)
        self.model        = model.to(self.device)
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.scheduler    = scheduler
        self.scaler = _create_grad_scaler(config, self.device)
        self.history: list[Dict[str, Any]] = []
        self.batch_adapter = batch_adapter or _DEFAULT_BATCH_ADAPTER
        self.callbacks = normalize_callbacks(callbacks=callbacks, logger=logger)
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: Optional[int] = None
        self.best_checkpoint_dir: Optional[Path] = None
        self.state = TrainerState(model=self.model,
                                  optimizer=self.optimizer,
                                  config=self.config,
                                  device=self.device,
                                  batch_adapter=self.batch_adapter,
                                  scheduler=self.scheduler,
                                  scaler=self.scaler,
                                  history=self.history,
                                 )
        if best_checkpoint_dir is not None:
            resolved = Path(best_checkpoint_dir).expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            self.best_checkpoint_dir = resolved

    ''' Function: fit
        Description: Train model for configured number of epochs with validation.
        Args:
            None
        Returns:
            List of dictionaries containing training history per epoch.
    '''
    def fit(self) -> list[Dict[str, Any]]:
        _notify_callbacks(self.callbacks, "on_run_start", self.state)
        patience = self.config.early_stopping_patience
        monitor_early_stop = (
            patience is not None and patience > 0 and self.val_loader is not None
        )
        lr_patience = self.config.lr_reduction_patience
        monitor_lr_reduction = (
            lr_patience is not None and lr_patience > 0 and self.val_loader is not None
        )
        lr_factor = self.config.lr_reduction_factor
        stagnant_epochs = 0
        stagnant_lr_epochs = 0

        try:
            for epoch in range(1, self.config.epochs + 1):
                self.state.epoch = epoch
                self.state.stage = "epoch"
                self.state.record = None
                self.state.metrics = None
                self.state.train_metrics = None
                self.state.val_metrics = None
                self.state.should_stop = False
                _notify_callbacks(self.callbacks, "on_epoch_start", self.state)

                train_metrics = train_one_epoch(self.model,
                                                self.train_loader,
                                                self.optimizer,
                                                self.loss_fn,
                                                self.config,
                                                scaler        = self.scaler,
                                                progress_desc = f"[train]",
                                                batch_adapter = self.batch_adapter,
                                                callbacks     = self.callbacks,
                                                state         = self.state,
                                               )

                val_metrics = None
                if self.val_loader is not None:
                    val_metrics = evaluate(self.model,
                                           self.val_loader,
                                           self.loss_fn,
                                           self.config.device,
                                           self.config.non_blocking,
                                           progress_desc = f"Epoch {epoch} [val]",
                                           batch_adapter = self.batch_adapter,
                                           callbacks     = self.callbacks,
                                           state         = self.state,
                                          )

                should_stop_after_epoch = False
                lr_reduced = False
                record_best = False
                best_checkpoint_path: Optional[Path] = None
                if val_metrics is not None:
                    current_loss = val_metrics.get("loss")
                    if current_loss is not None:
                        val_loss = float(current_loss)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            stagnant_epochs = 0
                            stagnant_lr_epochs = 0
                            self.best_state_dict = {key: tensor.detach().cpu().clone()
                                                    for key, tensor in self.model.state_dict().items()
                                                   }
                            self.best_epoch = epoch
                            record_best = True
                        else:
                            if monitor_early_stop:
                                stagnant_epochs += 1
                            if monitor_lr_reduction:
                                stagnant_lr_epochs += 1
                                if lr_patience is not None and stagnant_lr_epochs >= lr_patience:
                                    lr_reduced = self._reduce_learning_rate(lr_factor)
                                    stagnant_lr_epochs = 0
                            if monitor_early_stop and patience is not None and stagnant_epochs >= patience:
                                should_stop_after_epoch = True

                if record_best and self.best_checkpoint_dir is not None:
                    train_loss = _metric_value(train_metrics, "loss")
                    val_loss = _metric_value(val_metrics, "loss")
                    filename = _best_checkpoint_filename(epoch, train_loss, val_loss)
                    payload = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_state_dict": self.best_state_dict,
                    }
                    best_checkpoint_path = self.best_checkpoint_dir / filename
                    torch.save(payload, best_checkpoint_path)

                self._step_scheduler(val_metrics)
                record = {
                    "epoch" : epoch,
                    "train" : train_metrics,
                    "val"   : val_metrics,
                    "lr"    : self.optimizer.param_groups[0]["lr"],
                }
                if lr_reduced:
                    record["lr_reduced"] = True
                if record_best:
                    record["best_checkpoint"] = True
                if should_stop_after_epoch:
                    record["early_stop_triggered"] = True
                self.history.append(record)
                self.state.record = record
                self.state.metrics = record
                self.state.train_metrics = train_metrics
                self.state.val_metrics = val_metrics
                self.state.should_stop = should_stop_after_epoch
                if best_checkpoint_path is not None:
                    _notify_callbacks(self.callbacks, "on_checkpoint_saved", self.state, best_checkpoint_path)
                _notify_callbacks(self.callbacks, "on_epoch_end", self.state)
                if should_stop_after_epoch:
                    break
        finally:
            self.state.stage = "run_end"
            _notify_callbacks(self.callbacks, "on_run_end", self.state)

        return self.history

    ''' Function: _step_scheduler
        Description: Step learning rate scheduler based on validation metrics.
        Args:
            val_metrics : Optional validation metrics dictionary.
        Returns:
            None
    '''
    def _step_scheduler(self, val_metrics: Optional[Dict[str, float]]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_metrics is None:
                raise ValueError("Validation metrics required for ReduceLROnPlateau scheduler.")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()

    def _reduce_learning_rate(self, factor: float) -> bool:
        if factor <= 0 or factor >= 1:
            return False
        updated = False
        for group in self.optimizer.param_groups:
            current_lr = group.get("lr")
            if current_lr is None:
                continue
            group["lr"] = current_lr * factor
            updated = True
        return updated

    def best_model_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.best_state_dict is not None:
            return copy.deepcopy(self.best_state_dict)
        return self.model.state_dict()


''' Function: fit
    Description: Convenience function to train model using Trainer class.
    Args:
        model        : Neural network model to train.
        optimizer    : Optimizer for parameter updates.
        loss_fn      : Loss function for training.
        train_loader : Training data loader.
        config       : Training configuration.
        val_loader   : Optional validation data loader.
        scheduler    : Optional learning rate scheduler.
    Returns:
        List of dictionaries containing training history per epoch.
'''
def fit(model        : nn.Module,
        optimizer    : Optimizer,
        loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader : Iterable[Any],
        config       : TrainingConfig,
        val_loader   : Optional[Iterable[Any]] = None,
        scheduler    : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        logger       : Optional[Callable[[Dict[str, Any]], None]] = None,
        best_checkpoint_dir: Optional[Path | str] = None,
        batch_adapter: Optional[BatchAdapter] = None,
        callbacks    : Optional[Iterable[Callback]] = None,
       ) -> list[Dict[str, Any]]:
    trainer = Trainer(model        = model,
                      optimizer    = optimizer,
                      loss_fn      = loss_fn,
                      train_loader = train_loader,
                      config       = config,
                      val_loader   = val_loader,
                      scheduler    = scheduler,
                      logger       = logger,
                      best_checkpoint_dir = best_checkpoint_dir,
                      batch_adapter = batch_adapter,
                      callbacks     = callbacks,
                     )
    return trainer.fit()
