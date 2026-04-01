import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trainer_core import KeyedBatchAdapter, Trainer, TrainingConfig, evaluate


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


if __name__ == '__main__':
    unittest.main()
