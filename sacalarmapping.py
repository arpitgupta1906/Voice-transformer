import matplotlib.pyplot as plt
import numpy as np
# from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile


def plot(data):
    plt.plot(data, color='steelblue')


# Scalar multiplication by ratio of magnitudes
# in dono file ka naam dekhna hai code me sirf example leke btaya hai
FILENAME1 = "input1.wav"
FILENAME2 = "output1.wav"
FILENAME3 = "third1.wav"
rate1, wav_data1 = wavfile.read(FILENAME1)
rate2, wav_data2 = wavfile.read(FILENAME2)  # isko dekhna hai
rate3, wav_data3 = wavfile.read(FILENAME3)

# CALL SMOOTING

L1 = len(wav_data1)
f1 = np.linspace(0, 8000, L1)
x1 = np.fft.fft(wav_data1)
X1 = np.abs(x1) / L1  # magnitude of coefficient of FFT of source voice
L2 = len(wav_data2)
f2 = np.linspace(0, 8000, L2)
x2 = np.fft.fft(wav_data2)
X2 = np.abs(x2) / L2  # magnitude of  coefficient of FFT of target voice to be replicated
ratio = abs(x2) / abs(x1)  # ratio of coefficients r=d/c
Freq = np.linspace(0, 8000, L1)
plt.plot(Freq, ratio)  # plot ratio over double sided FFT spectrum

# This will help us cut down on the ratio .
# You need to pick values of h>1.
# The closer to 1 , the more it shrinks
# The farther h is from 1 , a the smaller the reduction

h = 1.25
Ratio = np.zeros(len(ratio))

for i in range(len(ratio)):
    Ratio[i] = ((h - 1) * ratio[i] + 1) / h
plt.plot(Freq.Ratio)

# Apply ratio multipilcation on third audio sample

L3 = len(wav_data3)
f3 = np.linspace(0, 8000, L3)
x3 = np.fft.fft(wav_data3)
x3 = x3 * Ratio

# Take IFFT of x3
