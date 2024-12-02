"""
Class for bounce datamodule
"""
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from dataset import SoundDS


class BounceDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, workers: int = 8):
        super().__init__()
        self.workers = workers
        self.batch_size = batch_size
        self.data_dir = Path("../data/sounds")
        self.train_csv = Path("../data/test.csv")
        self.val_csv = Path("../data/test.csv")

    def setup(self, stage: str):
        self.train_data = SoundDS(self.data_dir, self.train_csv, aug=False)
        self.val_data = SoundDS(self.data_dir, self.val_csv)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.workers
        )

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)
