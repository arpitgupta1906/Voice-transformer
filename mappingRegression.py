import matplotlib.pyplot as plt
import numpy as np
from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile


def plot(data):
    plt.plot(data, color='steelblue')


# function to smooth an audio signal
def smoothing(wav_data):
    r = 3
    wav = np.zeros(len(wav_data))
    for i in range(r, len(wav) - r):
        wav[i] = sum(wav_data[i - r:i + r + 1]) / (2 * r + 1)

    return wav


# function to return the regression coefficients
def regressionCoefficients(wav_data):
    # smoothen the signal
    wav1 = smoothing(wav_data)

    L1 = len(wav1)

    fft1 = np.linspace(0, 8000, L1)
    f1 = np.linspace(0, 4000, L1 / 2)

    xx1 = np.fft.fft(wav1)
    xx1 = np.abs(real(xx1)) / L1

    x1 = 2 * xx1[0:L1 / 2]

    max1 = max(x1)

    x1 = 2 * xx1[0:L1 / 2]
    max1 = max(x1)

    # find peaks above threshold of 20%
    peaks1 = findpeaks(x1, 50, .2)
    plt.plot(f1, x1)

    # plot peaks
    plt.plot(f1[peaks1], x1[peaks1], 'ro')
    plt.show()
    freq1 = f1[peaks1]

    # Regression Coefficients
    coef1 = x1[peaks1] * max1

    return coef1


# input in waves here
FILENAME1 = "output1.wav"
FILENAME2 = "output2.wav"
FILENAME3 = "output3.wav"

rate1, wav_data1 = wavfile.read(FILENAME1)
rate2, wav_data2 = wavfile.read(FILENAME2)
rate3, wav_data3 = wavfile.read(FILENAME3)

# coeficients of regression
coef1 = regressionCoefficients(wav_data1)
coef2 = regressionCoefficients(wav_data2)

# include end points of regression
n = len(coef1)
C1 = np.zeros(n + 2)
C1[1:n + 1] = coef1
C1[n + 1] = 1.2 * C1[n]
C2 = np.zeros(n + 2)
C2[1:n + 1] = coef2
C2[n + 1] = 1.2 * C2[n]

# plot coefficients
plt.plot(C1, C2, 'o')

# Construct polynomial of order n-1
X = np.asmatrix(C1)
Y = np.asmatrix(C2)
b = np.transpose(Y)

A = np.zeros([n, n])
A = np.asmatrix(A)

for i in range(0, n):
    for j in range(0, n):
        A[i, j] = X[0, i] ** (n - j - 1)
coef = A ** (-1) * b

# plot best fit line to coefficients
xaxis = np.linspace(0, C1[-1], 1000)
yaxis = np.linspace(0, 0, 1000)

for i in range(0, 1000):
    for j in range(n):
        yaxis[i] += coef[j] * xaxis[i] ** (n - j - 1)

plt.plot(X, Y, 'ro')
plt.plot(xaxis, yaxis)

# find FFT of third audio sample
L3 = len(wav_data3)
ff3 = np.linspace(0, 8000, L3)
f3 = np.linspace(0, 4000, L3 / 2)
xx3 = np.fft.fft(wav_data3)
xx3 = np.abs(real(xx3)) / L3
x3 = 2 * xx3[0:L3 / 2]

# push coefficients through regression model found from vector "coef"
n = len(coef)
ynew = np.linspace(0, 0, L3 / 2)
for i in range(L3 / 2):
    for j in range(n):
        ynew[i] += coef[j] * x3[i] ** (n - j - 1)

# plot the transformation of the coefficients
plt.plot(f3, yaxis)
