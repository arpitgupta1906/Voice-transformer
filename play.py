import pyaudio
import wave
from pylab import *

FILENAME1 = "output1.wav"
CHUNK = 1024

wf1 = wave.open(FILENAME1, 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf1.getsampwidth()), channels=wf1.getnchannels(),
                rate=wf1.getframerate(), output=True)

data1 = wf1.readframes(CHUNK)

while data1:
    stream.write(data1)
    data1 = wf1.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()
