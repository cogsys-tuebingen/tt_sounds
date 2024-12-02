"""
Class for bounce sound dataset and others
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from audio_utils import *
from plots import plot_spectrogram, plot_waveform

# Constants
DEFAULT_SR = 44100
DEFAULT_DURATION = 661  # duration in samples
DEFAULT_CHANNELS = 1
DEFAULT_SHIFT_PCT = 0.05


surface_classes = [
    "other",
    "racket_01",
    "racket_02",
    "racket_03",
    "racket_04",
    "racket_05",
    "racket_06",
    "racket_07",
    "racket_08",
    "racket_09",
    "racket_10",
    "table",
    "floor",
]
# spin_mag_class = ["no_spin", "with_spin"]
spin_classes = ["back", "none", "top"]


def convert_labels(attr):
    if attr["surface"] == "racket":
        surface_class = int(attr["racket-type"])
    else:
        surface_class = surface_classes.index(attr["surface"])
    spin_class = spin_classes.index(attr["spin-direction"])
    return surface_class, spin_class


class SoundDS(Dataset):
    def __init__(self, data_path, csv_path, aug=False, win_len=256):
        self.data_path = Path(data_path)
        self.labels = pd.read_csv(csv_path, sep=",")
        # print(self.labels.iloc[0]["surface"])
        # print(self.labels[0])
        self.duration = DEFAULT_DURATION
        self.sr = DEFAULT_SR
        self.channel = DEFAULT_CHANNELS
        self.shift_pct = DEFAULT_SHIFT_PCT
        self.win_len = win_len
        self.aug = aug

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.labels)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file
        audio_file = self.data_path / f"{self.labels['bounce-id'].iloc[idx]}.wav"
        # print(str(audio_file))

        surface_class, spin_class = convert_labels(self.labels.iloc[idx])

        # Load and preprocess the audio file
        aud = open_audio(audio_file)
        # plot_waveform(aud[0], self.sr)
        # plt.show()
        reaud = resample(aud, self.sr)
        rechan = rechannel(reaud, self.channel)
        dur_aud = pad_trunc(rechan, self.duration)
        if self.aug:
            shift_aud = time_shift(dur_aud, self.shift_pct)
            output = mel_spectro_gram(shift_aud, win_len=self.win_len)
            # output = spectro_augment(
            #     sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
            # )
        else:
            output = mel_spectro_gram(dur_aud, win_len=self.win_len)

        # plot_spectrogram(output)
        # plt.show()
        return (output, surface_class, spin_class)


if __name__ == "__main__":
    label_path = Path("../data/train.csv")
    data_path = Path("../data/sounds")

    dataset = SoundDS(data_path, label_path)
    print(len(dataset))
    spec1, surf1, spin1 = dataset[0]
    spec2, surf2, spin2 = dataset[1322]
    spec3, surf3, spin3 = dataset[3462]
    # print(spec1.shape)
    print(surf1)
    print(spin1)
    print(surf2)
    print(spin2)
    print(surf3)
    print(spin3)
    # f, axs = plt.subplots(3, 2)
    # plot_spectrogram(spec2[0], ax=axs[0, 0])
    # plot_spectrogram(spec1[0], ax=axs[1, 0])
    # plot_spectrogram(spec3[0], ax=axs[2, 0])

    # plot_spectrogram(np.abs(spec2[0] - spec1[0]), ax=axs[0, 1])
    # plot_spectrogram(np.abs(spec1[0] - spec1[0]), ax=axs[1, 1])
    # plot_spectrogram(np.abs(spec3[0] - spec1[0]), ax=axs[2, 1])

    f, axs = plt.subplots(3)
    plot_spectrogram(spec2[0], ax=axs[0])
    axs[0].set_title("Backpin")
    plot_spectrogram(spec1[0], ax=axs[1])
    axs[1].set_title("No Spin")
    plot_spectrogram(spec3[0], ax=axs[2])
    axs[2].set_title("Topspin")

    axs[0].set_xticks([])
    axs[1].set_xticks([])

    xlabels = [0] + list(np.linspace(0, 15, 6))
    axs[2].xaxis.set_ticklabels(xlabels)
    axs[2].set_xlabel("Time [ms]")

    plt.show()
