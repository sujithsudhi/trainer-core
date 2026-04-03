# trainer-core

Reusable PyTorch training engine intended to be shared across multiple local
repositories such as `Transformers` and `VLM`.

## Scope

This package currently contains the generic pieces only:

- training config loading
- batch adapters
- callback primitives
- generic train and eval loops
- a reusable `Trainer`

Task-specific metrics, plotting, dataset construction, and app-specific W&B
artifact logging should remain in the application repositories.

## Local Development

```powershell
cd C:\Users\Sujith\Dev\Repos\trainer-core
python -m pip install -e .
```

## Mixed Precision

Applications using this package can control AMP through `TrainingConfig`.

- `use_amp` enables or disables AMP. It only becomes active when training on CUDA.
- `amp_dtype` controls the autocast dtype.
- Accepted `amp_dtype` values are `auto`, `fp16`, `float16`, `bf16`, `bfloat16`,
  `torch.float16`, and `torch.bfloat16`.
- `auto` currently resolves to FP16.
- `GradScaler` is enabled for FP16 and disabled automatically for BF16.

Example:

```python
from trainer_core import TrainingConfig

config = TrainingConfig(
    device="cuda",
    use_amp=True,
    amp_dtype="bf16",
)
```

If you load config from a mapping or JSON payload:

```python
from trainer_core import load_training_config

config = load_training_config(
    {
        "device": "cuda",
        "use_amp": True,
        "amp_dtype": "fp16",
    }
)
```
