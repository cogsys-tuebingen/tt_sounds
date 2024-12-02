"""
Script to run validation of the classification model
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

if __name__ == "__main__":
    model_path = "/home/tg/Git/tt_sound/src/tt_sound/d9lfja44/checkpoints/"

    model = AudioClassifier.load_from_checkpoint(checkpoint_path=model_path)
