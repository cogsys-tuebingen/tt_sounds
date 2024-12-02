"""
Functions to handle audio data
"""

import random

import torch
import torchaudio
from torchaudio import transforms


# ----------------------------
# Load an audio file. Return the signal as a tensor and the sample rate
# ----------------------------
def open_audio(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return sig, sr


# ----------------------------
# Convert the given audio to the desired number of channels
# ----------------------------
def rechannel(aud, new_channel):
    sig, sr = aud

    if sig.shape[0] == new_channel:
        return aud

    if new_channel == 1:
        resig = sig[:1, :]
    else:
        resig = torch.cat([sig, sig])

    return resig, sr


# ----------------------------
# Resample the audio to a new sample rate
# ----------------------------
def resample(aud, newsr):
    sig, sr = aud

    if sr == newsr:
        return aud

    num_channels = sig.shape[0]
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

    if num_channels > 1:
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
        resig = torch.cat([resig, retwo])

    return resig, newsr


# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_len' in samples
# ----------------------------
def pad_trunc(aud, max_len):
    sig, sr = aud
    num_rows, sig_len = sig.shape

    if sig_len > max_len:
        sig = sig[:, :max_len]
    elif sig_len < max_len:
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return sig, sr


# ----------------------------
# Shift the signal in time
# ----------------------------
def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return sig.roll(shift_amt), sr


# ----------------------------
# Generate a Spectrogram
# ----------------------------
def spectro_gram(aud, n_fft=1024, win_len=128, normalized=True):
    sig, sr = aud
    top_db = 80

    spec = transforms.Spectrogram(
        n_fft=n_fft, win_length=win_len, hop_length=win_len // 2, normalized=normalized
    )(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


# ----------------------------
# Generate a Mel Spectrogram
# ----------------------------
def mel_spectro_gram(aud, n_mels=64, n_fft=1024, win_len=128):
    sig, sr = aud
    top_db = 80

    spec = transforms.MelSpectrogram(
        sr, n_fft=n_fft, win_length=win_len, hop_length=win_len // 2, n_mels=n_mels
    )(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


# ----------------------------
# Generate a MFCC
# ----------------------------
def mfcc(aud, n_fft=1024, win_len=128, n_mels=64, n_mfcc=64):
    sig, sr = aud
    top_db = 80

    mfcc_transform = transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "win_length": win_len,
            "hop_length": win_len // 2,
            "mel_scale": "htk",
        },
    )

    mfcc = mfcc_transform(sig)
    mfcc = transforms.AmplitudeToDB(top_db=top_db)(mfcc)
    return mfcc


# ----------------------------
# Augment the Spectrogram by masking
# ----------------------------
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec
