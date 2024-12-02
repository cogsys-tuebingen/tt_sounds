import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
import numpy as np
import matplotlib.pylab as plt
import scipy.signal
import pandas as pd
from collections import deque

class Detector():
    def __init__(self) -> None:
        pass

    def printBounces(self, frames: list, probabilities: list, energies: list, averages: list, thresholds: list, file) -> None:
        predictions = [frame for frame, prob in zip(frames, probabilities) if prob == 1]
        frames = [frame/1000 for frame in frames]

        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)
        waveform = SPEECH_WAVEFORM.numpy()

        _, ax = plt.subplots(1, 1)
        librosa.display.waveshow(y=waveform[0], sr=SAMPLE_RATE, axis='s', ax=ax)
        ax.grid(True)
        ax.set_title("Bounce detection")

        predictions = [ms/1000 for ms in predictions]

        ax.scatter(predictions, [0 for _ in range(len(predictions))], color='yellow', marker='x', zorder=10, label='predictions')
        ax.plot(frames, energies, color='black', label='Frame Energy')
        ax.plot(frames, averages, color='orange', label='Energy Average')
        ax.plot(frames, thresholds, color='red', label='Detection Threshold')

        plt.xlabel("Time in s")
        plt.ylabel("Amplitude in V")
        plt.legend()
        plt.show()

    def findBounces(self, file, detector, kwargs={}) -> None:
        frames, probabilities, energies, averages, thresholds = detector(file, kwargs=kwargs)
        self.printBounces(frames, probabilities, energies, averages, thresholds, file)

    def AdjacentFrameDetect(self, file: str, kwargs={"threshold": 0.005}):
        # important parameters and variables
        frameLength = 1 # in ms; frame energy must peak (phase 1)
        absoluteFrameThreshold = kwargs["threshold"] # amount of energy a frame must surpass the previous frame by
        startSample = 0 # start sample of frame
        previousFrameEnergy = 0
        timeoutLength = 20 # in ms; period where no bounce can be detected after any detected bounce

        # internal definitions
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)
        SPEECH_WAVEFORM_ABS = np.abs(SPEECH_WAVEFORM)
        totalDuration = len(SPEECH_WAVEFORM[0])/SAMPLE_RATE*1000 # in ms
        frames = [ms for ms in range(frameLength, int(totalDuration), frameLength)]
        probabilities = [0 for _ in range(frameLength, int(totalDuration), frameLength)]
        energies = [0 for _ in range(frameLength, int(totalDuration), frameLength)]
        averages = [0 for _ in range(len(frames))]
        thresholds = [0 for _ in range(frameLength, int(totalDuration), frameLength)]
        timeout = 0

        for idms, ms in enumerate(frames):
            # get energy of current frame
            endSample = int(SAMPLE_RATE/1000*ms)
            frame = SPEECH_WAVEFORM_ABS[0][startSample:endSample]
            energy = torch.mean(frame)

            energies[idms] = energy
            thresholds[idms] = previousFrameEnergy + absoluteFrameThreshold

            # timeout if last bounce was < timeoutLength long
            if timeout > 0:
                timeout -= 1
            else:
                # energy spike test
                if previousFrameEnergy + absoluteFrameThreshold < energy:
                    probabilities[idms] = 1
                    timeout = timeoutLength

            previousFrameEnergy = energy
            startSample = endSample

        return frames, probabilities, energies, averages, thresholds
    
    def MovingAverageDetect(self, file: str, kwargs={"threshold": 5}):
        # important parameters and variables
        frameLength = 1 # in ms; frame energy must peak (phase 1)
        relativeAverageThreshold = kwargs["threshold"] # multiplier: how much bigger than the average is a peak?
        startSample = 0 # start sample of frame
        timeoutLength = 50 # in ms; period where no bounce can be detected after any detected bounce
        energySize = 100 # max size of queue for moving average

        # internal definitions
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)
        SPEECH_WAVEFORM_ABS = np.abs(SPEECH_WAVEFORM)
        totalDuration = len(SPEECH_WAVEFORM[0])/SAMPLE_RATE*1000 # in ms
        frames = [ms for ms in range(frameLength, int(totalDuration), frameLength)]
        probabilities = [0 for _ in range(len(frames))]
        energies = [0 for _ in range(len(frames))]
        averages = [0 for _ in range(len(frames))]
        thresholds = [0 for _ in range(len(frames))]
        averageEnergy = 0
        pastEnergies = deque([], maxlen=energySize)
        timeout = 0

        for idms, ms in enumerate(frames):
            endSample = int(SAMPLE_RATE/1000*ms)

            # get energy of current frame
            frame = SPEECH_WAVEFORM_ABS[0][startSample:endSample]
            energy = torch.mean(frame)

            # init averageEnergy as first frame energy
            if ms == frames[0]:
                averageEnergy = energy

            energies[idms] = energy.item()
            averages[idms] = averageEnergy.item()
            thresholds[idms] = averageEnergy.item() * relativeAverageThreshold

            # timeout if last bounce was < timeoutLength long
            if timeout > 0:
                timeout -= 1
            else:
                # energy spike test
                if averageEnergy * relativeAverageThreshold < energy:
                    probabilities[idms] = 1
                    timeout = timeoutLength

            # calculate new average
            pastEnergies.append(energy)
            averageEnergy = np.mean(pastEnergies)
            startSample = endSample

        return frames, probabilities, energies, averages, thresholds

    def DecayAverageDetect(self, file: str, kwargs={"threshold": 3, "decay": 0.9}):
        # important parameters and variables
        frameLength = 1 # in ms; frame energy must peak (phase 1)
        relativeAverageThreshold = kwargs["threshold"] # multiplier: how much bigger than the average is a peak?
        startSample = 0 # start sample of frame
        decay = 1 - kwargs["decay"] # how big of an impact do energies make to the current average energy?
        timeoutLength = 100 # in ms; period where no bounce can be detected after any detected bounce

        # internal definitions
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)
        SPEECH_WAVEFORM_ABS = np.abs(SPEECH_WAVEFORM)
        totalDuration = len(SPEECH_WAVEFORM[0])/SAMPLE_RATE*1000 # in ms
        frames = [ms for ms in range(frameLength, int(totalDuration), frameLength)]
        probabilities = [0 for _ in range(len(frames))]
        energies = [0 for _ in range(len(frames))]
        averages = [0 for _ in range(len(frames))]
        thresholds = [0 for _ in range(len(frames))]
        averageEnergy = 0
        timeout = 0

        for idms, ms in enumerate(frames):
            endSample = int(SAMPLE_RATE/1000*ms)

            # get energy of current frame
            frame = SPEECH_WAVEFORM_ABS[0][startSample:endSample]
            energy = torch.mean(frame)

            # init averageEnergy as first frame energy
            if ms == frames[0]:
                averageEnergy = energy

            energies[idms] = energy.item()
            averages[idms] = averageEnergy.item()
            thresholds[idms] = averageEnergy.item() * relativeAverageThreshold

            # timeout if last bounce was < timeoutLength long
            if timeout > 0:
                timeout -= 1
            else:
                # energy spike test
                if averageEnergy * relativeAverageThreshold < energy:
                    probabilities[idms] = 1
                    timeout = timeoutLength

            # calculate new average
            energyDiff = energy - averageEnergy
            averageEnergy = averageEnergy + energyDiff * decay
            startSample = endSample

        return frames, probabilities, energies, averages, thresholds
    
    def scipyPeaks(self, file: str, kwargs={}):
        frameLength = 1 # in ms; frame energy must peak (phase 1)
        timeoutLength = 100 # in ms; period where no bounce can be detected after any detected bounce

        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(file)
        SPEECH_WAVEFORM_ABS = np.abs(SPEECH_WAVEFORM)
        totalDuration = len(SPEECH_WAVEFORM[0])/SAMPLE_RATE*1000+1 # in ms
        frames = [ms for ms in range(frameLength, int(totalDuration), frameLength)]
        probabilities = [0 for _ in range(frameLength, int(totalDuration), frameLength)]

        predictions = scipy.signal.find_peaks(SPEECH_WAVEFORM_ABS[0], distance=SAMPLE_RATE/1000*timeoutLength)
        print(len(predictions[0]), len(probabilities))
        for id, pred in enumerate(predictions[0]):
            print(id, pred // (SAMPLE_RATE // 1000 * frameLength) + 1)
            probabilities[pred // (SAMPLE_RATE // 1000 * frameLength)] = 1

        return frames, probabilities

    def autoLabel(self, soundFile: str, labelFile: str, detector, kwargs={}):
        # get soundfile info
        df = pd.read_csv('data-info.csv', sep=';')
        df = df.loc[df["file"] == soundFile]
        surface = df.iloc[0]["surface"]
        spinDir = df.iloc[0]["spin-direction"]
        spinMag = df.iloc[0]["spin-magnitude"]
        racketType = df.iloc[0]["racket-type"]
        
        # get predictions
        path = "./audio_data/" + soundFile
        if kwargs:
            frames, probabilities, energies, averages, thresholds = detector(path, kwargs=kwargs)
        else:
            frames, probabilities, energies, averages, thresholds = detector(path)
        predictions = [frame for frame, prob in zip(frames, probabilities) if prob == 1]

        # get last label id
        prevLabels = pd.read_csv(labelFile, sep=';')
        bounceID = prevLabels.iloc[-1]["bounce-id"]

        # save labels
        with open(labelFile, "a") as labels:
            for pred in predictions:
                bounceID = bounceID + 1
                timestamp = str((pred - 1) / 1000) # round down for labeling
                labels.write(f"{bounceID};{soundFile};{timestamp};{surface};{racketType};{spinMag};{spinDir}\n")
        
        self.printBounces(frames, probabilities, energies, averages, thresholds, path)

    def labelFromFile(self, timestampFile: str, labelFile: str, label_details: dict, surfacesGiven=False):
        """
        surfacesGiven: set true if surface information is in timestampFile; else give information in label_details
        """
        # get soundfile info
        soundFile = label_details["file"]
        if not surfacesGiven:
            surface = label_details["surface"]
        racketType = label_details["racket-type"]
        spinMag = label_details["spin-magnitude"]
        spinDir = label_details["spin-direction"]

        # get predictions
        df = pd.read_csv(timestampFile, sep='\t', on_bad_lines='warn')
        print(df.head())
        timestamps = df["time"]
        print(timestamps)

        # if multiple surfaces in timestampFile:
        if surfacesGiven:
            surface_mapping = {"r": "racket", "t": "table", "f": "floor"}
            type_mapping = {"r": racketType, "t": "none", "f": "none"}
            magnitude_mapping = {"r": spinMag, "t": "none", "f": "none"}
            direction_mapping = {"r": spinDir, "t": "none", "f": "none"}
            surfaces = df["surface"]
            types = [type_mapping[surf] for surf in surfaces]
            magnitudes = [magnitude_mapping[surf] for surf in surfaces]
            directions = [direction_mapping[surf] for surf in surfaces]
            surfaces = [surface_mapping[surf] for surf in surfaces]

        # get last label id
        prevLabels = pd.read_csv(labelFile, sep=';')
        bounceID = prevLabels.iloc[-1]["bounce-id"]
        
        # save labels
        with open(labelFile, "a") as labels:
            for id, timestamp in enumerate(timestamps):
                bounceID = bounceID + 1
                if surfacesGiven:
                    surface = surfaces[id]
                    racketType = types[id]
                    spinMag = magnitudes[id]
                    spinDir = directions[id]
                labels.write(f"{bounceID};{soundFile};{timestamp};{surface};{racketType};{spinMag};{spinDir}\n")


    def detect(self):
        """
        deprecated
        """

        filename = "./audio_data/STE-015.wav"
        y, sr = librosa.load(filename, sr=None)
        n_fft = 2048
        hop_length = 512

        amps = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.core.fft_frequencies(sr=sr)
        times = librosa.core.frames_to_time(amps[0], sr=sr, n_fft=n_fft, hop_length=hop_length)
        # MAX BIN NEEDS TO BE 41-44
        fft_bin = 44 # 947 Hz
        print(freqs[fft_bin])
        print(f'freqs: {len(freqs)}, times: {len(times)}')
        print(f"amp range: [{round(np.min(amps), 6)};{round(np.max(amps), 6)}]")
        audio_length = librosa.get_duration(y=y, sr=sr)
        print(audio_length)

        maxBin = [0] * len(times)

        # print('freq (Hz)', freqs[fft_bin])
        # print('time (s)', times)
        # print('amplitude', amps[fft_bin, time_idx])
        timestamps = [audio_length / len(times) * tid for tid in range(len(times))]
        probability = [0] * len(times)
        for time_idx in range(len(times)):
            if amps[fft_bin, time_idx] > 2:
                probability[time_idx] = 1
            for bin in range(len(freqs)):
                if amps[bin, time_idx] > amps[maxBin[time_idx], time_idx]:
                    maxBin[time_idx] = bin
            # print(f'time: {timestamp}, freq: {freqs[fft_bin]}, amp: {amps[fft_bin, time_idx]}')
        print(maxBin)
        plt.plot(timestamps, probability)
        plt.show()

    def freqchecker(self):
        filename = "./audio_data/STE-015.wav"
        y, sr = librosa.load(filename, sr=None)
        
        amps = np.abs(librosa.stft(y))
        dbs = librosa.amplitude_to_db(amps, ref=np.max)
        freqs = librosa.core.fft_frequencies(sr=sr)
        times = librosa.core.frames_to_time(amps[0], sr=sr)

        audio_length = librosa.get_duration(y=y, sr=sr)
        n = 7
        max_freqs = [0] * n
        max_bins = [0] * n
        max_amps = [0] * n

        for i in range(n):
            for bin, freq in enumerate(freqs):
                print(f"window: {i}, frequency: {freq}, amplitude: {amps[bin, i]}")
                if amps[bin, i] > max_amps[i]:
                    max_freqs[i] = freq
                    max_bins[i] = bin
                    max_amps[i] = amps[bin, i]
        
        print(max_freqs)
        print(max_bins)
        print(sr, len(freqs), len(amps))

# detector = Detector()
# Detector.detect(Detector)
# Detector.freqchecker(Detector)