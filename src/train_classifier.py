"""
Script to train the classifier
"""
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI
# from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from classifier import AudioClassifier
from datamodule import BounceDataModule
from dataset import SoundDS, surface_classes  # Adjust this import if needed,


def main():
    cli = LightningCLI(
        model_class=AudioClassifier,
        datamodule_class=BounceDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    seed_everything(42, workers=True)
    main()
