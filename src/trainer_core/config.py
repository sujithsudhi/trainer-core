"""Thin config exports for the starter trainer-core package."""

from .engine import TrainingConfig, load_training_config

TrainerConfig       = TrainingConfig
load_trainer_config = load_training_config

__all__ = [
    "TrainingConfig",
    "TrainerConfig",
    "load_training_config",
    "load_trainer_config",
]
