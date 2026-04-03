import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainer_core import KeyedBatchAdapter, Trainer, TrainingConfig, evaluate, load_training_config
from trainer_core.engine import _is_grad_scaling_enabled


class DictDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'image': torch.randn(4), 'labels': torch.randn(1)}


class EngineSmokeTests(unittest.TestCase):
    def test_trainer_fit_runs(self):
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        model = nn.Linear(4, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = TrainingConfig(epochs=2, device='cpu', use_amp=False)
        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=nn.MSELoss(), train_loader=loader, val_loader=loader, config=config)
        history = trainer.fit()
        self.assertEqual(len(history), 2)

    def test_keyed_batch_adapter_runs(self):
        loader = DataLoader(DictDataset(), batch_size=2)
        metrics = evaluate(model=nn.Linear(4, 1), dataloader=loader, loss_fn=nn.MSELoss(), device='cpu', batch_adapter=KeyedBatchAdapter('image', 'labels'))
        self.assertIn('loss', metrics)

    def test_training_config_resolves_amp_dtype(self):
        self.assertEqual(TrainingConfig(amp_dtype='fp16').amp_dtype, 'float16')
        self.assertEqual(TrainingConfig(amp_dtype='bf16').amp_dtype, 'bfloat16')
        self.assertEqual(TrainingConfig(amp_dtype=torch.float16).amp_dtype, 'float16')
        self.assertEqual(TrainingConfig(amp_dtype=torch.bfloat16).amp_dtype, 'bfloat16')

    def test_load_training_config_accepts_amp_dtype(self):
        config = load_training_config({'device': 'cpu', 'use_amp': False, 'amp_dtype': 'bf16'})
        self.assertEqual(config.amp_dtype, 'bfloat16')

    def test_invalid_amp_dtype_raises(self):
        with self.assertRaises(ValueError):
            TrainingConfig(amp_dtype='fp32')

    def test_bf16_disables_grad_scaling(self):
        with patch('trainer_core.engine.torch.cuda.is_available', return_value=True):
            bf16_config = TrainingConfig(device='cuda', use_amp=True, amp_dtype='bf16')
            fp16_config = TrainingConfig(device='cuda', use_amp=True, amp_dtype='fp16')
            self.assertFalse(_is_grad_scaling_enabled(bf16_config, torch.device('cuda')))
            self.assertTrue(_is_grad_scaling_enabled(fp16_config, torch.device('cuda')))


if __name__ == '__main__':
    unittest.main()
