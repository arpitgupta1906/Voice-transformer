import matplotlib.pyplot as plt
import numpy as np
from findpeaks import findpeaks
import pyaudio
import wave
from smoothing import smoothing
from pylab import *
from scipy.io import wavfile
from shiftallpeaks import shiftallpeaks


FILENAME1 = "source.wav"
FILENAME2 = "map.wav"
FILENAME3 = "target.wav"

rate1, wav_data1 = wavfile.read(FILENAME1)
rate2, wav_data2 = wavfile.read(FILENAME2)
rate3, wav_data3 = wavfile.read(FILENAME3)

wav1 = smoothing(wav_data1)
wav2 = smoothing(wav_data2)
wav3 = smoothing(wav_data3)

wav1=np.fft.fft(wav1)
wav2=np.fft.fft(wav2)

peaks1 = findpeaks(wav1, 50, 0.2)

peaks2= findpeaks(wav2, 50, 0.2)

new=shiftallpeaks(peaks1,peaks2,wav3)
new=np.fft.ifft(new)

plt.plot(new)
plt.show()

