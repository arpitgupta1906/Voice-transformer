import matplotlib.pyplot as plt
import numpy as np
#from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile

def plot(data):
    plt.plot(data, color ='steelblue')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 4

FILENAME1 = "output1.wav"

p = pyaudio.PyAudio()

stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input = True, frames_per_buffer = CHUNK)

print("*_recording")

frames = []

for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("*_done_recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(FILENAME1, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

wf1 = wave.open(FILENAME1, 'rb')

p = pyaudio.PyAudio()

stream = p.open(format = p.get_format_from_width(wf1.getsampwidth()), channels = wf1.getnchannels(), rate = wf1.getframerate(), output= True)

data1 = wf1.readframes(CHUNK)

while data1:
    stream.write(data1)
    data1 = wf1.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()


spf = wave.open(FILENAME1, 'r')
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
if spf.getnchannels() == 2:
    print ('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()

