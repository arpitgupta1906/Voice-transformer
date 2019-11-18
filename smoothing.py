import numpy as np


def smoothing(wav_data):
    r = 3
    wav = np.zeros(len(wav_data))
    for i in range(r, len(wav) - r):
        wav[i] = sum(wav_data[i - r:i + r + 1]) / (2 * r + 1)

    return wav
