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
