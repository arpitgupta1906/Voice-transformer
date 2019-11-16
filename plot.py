import matplotlib.pyplot as plt
import numpy as np
# from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile

FILENAME1 = "output1.wav"


spf = wave.open(FILENAME1, 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
if spf.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()
