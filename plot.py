import matplotlib.pyplot as plt
import numpy as np
# from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile

FILENAME1 = "output1.wav"

rate, data = wavfile.read(FILENAME1)

plt.plot(data)
plt.show()