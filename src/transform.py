import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import os
import soundfile as sf
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from glob import glob

import librosa
import librosa.display
import scipy.signal

class LibrosaTransform():
    def __init__(self) -> None:
        pass

    def create_spectrograms(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('spectograms', file + '.png') for file in filenames]

        for idf, file in enumerate(audio_paths):
            y, sr = librosa.load(file, sr=None)

            D = librosa.stft(y, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Plot the transformed audio data
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(S_db,
                                        sr=sr,
                                        x_axis='time',
                                        y_axis='log',
                                        ax=ax)
            ax.set_title('Spectogram ' + str(idf), fontsize=20)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.savefig(video_paths[idf])
            plt.close()

    def create_mel_spectrograms(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('mel_spectograms', file + '.png') for file in filenames]

        for idf, file in enumerate(audio_paths):
            y, sr = librosa.load(file, sr=None)

            S = librosa.feature.melspectrogram(y=y,
                                    sr=sr)
            S_db = librosa.amplitude_to_db(S, ref=np.max)

            # Plot the transformed audio data
            fig, ax = plt.subplots(figsize=(10, 5))
            # Plot the mel spectogram
            img = librosa.display.specshow(S_db,
                                            sr=sr,
                                            x_axis='time',
                                            y_axis='mel',
                                            ax=ax)
            ax.set_title('Mel Spectogram ' + str(idf), fontsize=20)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.savefig(video_paths[idf])
            plt.close()

    def create_waveforms(self, audio_files='audio_data', files='*.wav'):
        sns.set_theme(style="white", palette=None)
        color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('waveforms', file + '.png') for file in filenames]

        for idf, file in enumerate(audio_paths):
            y, sr = librosa.load(file, sr=None)

            _, ax = plt.subplots(figsize=(10, 5))

            img = librosa.display.waveshow(y=y,
                                    sr=sr,
                                    axis='time',
                                    ax=ax)
            ax.set_title('Waveform ' + str(idf), fontsize=20)

            plt.savefig(video_paths[idf])
            plt.close()
            print(f"Waveform {file} done")

    def create_file_mfccs(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('mfccs', file + '.png') for file in filenames]

        for idf, file in enumerate(audio_paths):
            y, sr = librosa.load(file, sr=None)

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # Plot the transformed audio data
            fig, ax = plt.subplots(figsize=(10, 5))
            img = librosa.display.specshow(mfccs,
                                        sr=sr,
                                        x_axis='time',
                                        y_axis='log',
                                        ax=ax)
            ax.set_title('MFCC ' + str(idf), fontsize=20)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            plt.savefig(video_paths[idf])
            plt.close()

    # high pass filter
    def highpass(self, data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data

    def preprocess(self, freq: int, audio_files='raw_data', files='*.wav'):
        print(f"Starting preprocessing of {audio_files}")
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path) for path in audio_paths]
        # print(filenames)
        video_paths = [os.path.join('audio_data', file) for file in filenames]

        for idf, file in enumerate(audio_paths):
            # load data
            y, sr = librosa.load(file, sr=None)
            # print(sr)
            # print(sr)
            # trim edges
            # y, _ = librosa.effects.trim(y, top_db=20)

            # apply high pass filter
            if freq > 0:
                y = self.highpass(y, freq, sr)

            # finish
            sf.write(video_paths[idf], y, sr)
        print(f"Finished preprocessing of {audio_files}")

class TorchTransform():
    def __init__(self) -> None:
        pass

    def create_waveforms(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('waveforms', file + '.png') for file in filenames]

        for idf, file in enumerate(audio_paths):
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)

            waveform = SPEECH_WAVEFORM.numpy()
    
            _, num_frames = waveform.shape
            time_axis = torch.arange(0, num_frames) / SAMPLE_RATE

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1)
            ax.plot(time_axis, waveform[0], linewidth=1)
            ax.grid(True)
            ax.set_xlim([0, time_axis[-1]])
            ax.set_title('Waveform ' + str(idf), fontsize=20)

            plt.xlabel("Time in s")
            plt.ylabel("Amplitude in dB")

            plt.savefig(video_paths[idf])
            plt.close()
            print(f"Waveform {file} completed")

    def create_spectrograms(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('spectograms', file + '.png') for file in filenames]

        spectrogram = T.Spectrogram(n_fft=512)
        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        for idf, file in enumerate(audio_paths):
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)

            spec = spectrogram(SPEECH_WAVEFORM)

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1)
            ax.imshow(librosa.power_to_db(spec[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_title('Spectogram ' + str(idf), fontsize=20)
            ax.set_ylabel("freq_bin")

            plt.savefig(video_paths[idf])
            plt.close()
            print(f"Spectrogram {file} completed")

    def create_mel_spectrograms(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('mel_spectograms', file + '.png') for file in filenames]

        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        for idf, file in enumerate(audio_paths):
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)

            mel_spectrogram = T.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                n_mels=n_mels,
                mel_scale="htk",
            )

            melspec = mel_spectrogram(SPEECH_WAVEFORM)

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1)
            ax.imshow(librosa.power_to_db(melspec[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_title('Mel Spectogram ' + str(idf), fontsize=20)
            ax.set_ylabel("freq_bin")

            plt.savefig(video_paths[idf])
            plt.close()
            print(f"Mel Spectrogram {file} completed")

    def create_file_mfccs(self, audio_files='audio_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path)[:-4] for path in audio_paths]
        video_paths = [os.path.join('mfccs', file + '.png') for file in filenames]

        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 256

        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        for idf, file in enumerate(audio_paths):
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)

            mfcc_transform = T.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )

            mfcc = mfcc_transform(SPEECH_WAVEFORM)

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1)
            ax.imshow(librosa.power_to_db(mfcc[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_title('MFCC ' + str(idf), fontsize=20)
            ax.set_ylabel("freq_bin")

            plt.savefig(video_paths[idf])
            plt.close()
            print(f"MFCC {file} completed")

    # high pass filter
    def highpass(self, data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data

    def preprocess(self, freq: int, audio_files='raw_data', files='*.wav'):
        audio_paths = glob(os.path.join(audio_files, files))
        filenames = [os.path.basename(path) for path in audio_paths]
        # print(filenames)
        video_paths = [os.path.join('audio_data', file) for file in filenames]

        for idf, file in enumerate(audio_paths):
            # load data
            y, sr = librosa.load(file, sr=None)
            # print(sr)
            # trim edges
            # y, _ = librosa.effects.trim(y, top_db=20)

            # apply high pass filter
            if freq > 0:
                y = self.highpass(y, freq, sr)

            # finish
            sf.write(video_paths[idf], y, sr)

class Preprocess():
    def __init__(self) -> None:
        pass

    def getBounces(self, labelFile="handLabels.csv", offset=3, duration=15) -> pd.DataFrame:
        """
        Collect data from given labeled bounces, add them to the labels DataFrame
        """
        labels = pd.read_csv(labelFile, sep=';')
        labels["path"] = "./audio_data/" + labels["original-file"]

        waveforms = []
        srs = []

        for idr in range(len(labels)):
            bounce = labels.loc[idr]
            frame_offset = int(44100*(bounce["timestamp"]-offset/1000))
            num_frames = int(44100*(duration/1000))
            SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(bounce["path"], frame_offset=frame_offset, num_frames=num_frames)

            if SAMPLE_RATE != 44100:
                path = bounce["path"]
                print(f"Sample Rate of {path} is not 44100: frames will be adjusted (but should still work as usual)")
                frame_offset = int(SAMPLE_RATE*(bounce["timestamp"]-offset/1000))
                num_frames = int(SAMPLE_RATE*(duration/1000))
                SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(bounce["path"], frame_offset=frame_offset, num_frames=num_frames)

            waveforms.append(SPEECH_WAVEFORM)
            srs.append(SAMPLE_RATE)
        
        labels["series"] = waveforms
        labels["sr"] = srs

        return labels 

    def saveAudios(self, bounces) -> None:
        print("Saving bounce audios")
        for idr in range(len(bounces)):
            bounce = bounces.loc[idr]
            torchaudio.save("./bounce_audios/" + str(bounce["bounce-id"]) + ".wav", bounce["series"], bounce["sr"])
        print("Bounce audios complete")
        return

    def saveWaveforms(self, bounces) -> None:
        """
        Create Waveforms for given bounces
        """
        print("Saving bounce waveforms")

        for idr in range(len(bounces)):
            bounce = bounces.loc[idr]

            waveform = bounce["series"].numpy()
    
            _, num_frames = waveform.shape
            time_axis = torch.arange(0, num_frames) / bounce["sr"]

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1, figsize=(10,5))
            ax.plot(time_axis, waveform[0], linewidth=1)
            ax.set_xlabel("Time (s)", fontdict={"fontsize": "x-large"})
            ax.set_ylabel("Amplitude (V)", fontdict={"fontsize": "x-large"})
            ax.grid(True)
            bounceID = bounce["bounce-id"]
            # ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"./bounce_waveforms/{bounceID}.png")
            plt.close()
        
        print("Bounce waveforms complete")
        return

    def saveMFCCs(self, bounces):
        """
        Create MFCCs for given bounces
        """
        n_fft = 1024
        win_length = 128
        hop_length = win_length//2
        n_mels = 64
        n_mfcc = 64

        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        print("Saving bounce MFCCs")

        for idr in range(len(bounces)):
            bounce = bounces.loc[idr]

            mfcc_transform = T.MFCC(
                sample_rate=bounce["sr"],
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "win_length": win_length,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )

            mfcc = mfcc_transform(bounce["series"])

            # Plot the transformed audio data
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
            cax = ax.imshow(librosa.power_to_db(mfcc[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical', label='Intensity (dB)')
            cbar.ax.yaxis.label.set_size(14)
            ax.set_xlabel("Time Frame Windows", fontdict={"fontsize": "x-large"})
            ax.set_ylabel("MFCC Cepstral Coefficients", fontdict={"fontsize": "x-large"})
            bounceID = bounce["bounce-id"]
            # ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"./bounce_mfccs/{bounceID}.png")
            plt.close()
        
        print("Bounce MFCCs complete")
        return
    
    def saveMels(self, bounces):
        """
        Create MEL Spectrograms for given bounces
        """
        n_fft = 1024
        win_length = 128
        hop_length = win_length//2
        n_mels = 64

        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        print("Saving bounce Mels")

        for idr in range(len(bounces)):
            bounce = bounces.loc[idr]

            mel_transform = T.MelSpectrogram(
                sample_rate=bounce['sr'],
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                n_mels=n_mels,
                mel_scale="htk",
            )

            mel = mel_transform(bounce["series"])

            # Plot the transformed audio data
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
            cax = ax.imshow(librosa.power_to_db(mel[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical', label='Intensity (dB)')
            cbar.ax.yaxis.label.set_size(14)
            ax.set_xlabel("Time Frame Windows", fontdict={"fontsize": "x-large"})
            ax.set_ylabel("Mel Frequency Bins", fontdict={"fontsize": "x-large"})
            bounceID = bounce["bounce-id"]
            # ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"./bounce_mels/{bounceID}.png")
            plt.close()
        
        print("Bounce Mels complete")
        return
    
    def saveSpectros(self, bounces):
        """
        Create MEL Spectrograms for given bounces
        """
        n_fft = 1024
        win_length = 128
        hop_length = win_length//2

        cmap = plt.get_cmap('viridis')
        cmap.set_under(color='k', alpha=None)

        print("Saving bounce Spectrograms")

        spectro_transform = T.Spectrogram(n_fft=n_fft, power=2, win_length=win_length, hop_length=hop_length)

        for idr in range(len(bounces)):
            bounce = bounces.loc[idr]

            spec = spectro_transform(bounce["series"])

            # Plot the transformed audio data
            _, ax = plt.subplots(1, 1)
            ax.imshow(librosa.power_to_db(spec[0]), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap)
            bounceID = bounce["bounce-id"]
            ax.axis('off')

            plt.savefig(f"./bounce_spectrograms/{bounceID}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
        
        print("Bounce Spectrograms complete")
        return