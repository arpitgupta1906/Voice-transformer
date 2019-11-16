# from scipy.fftpack import fft,fftfreq
from cmath import exp, pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.compat import integer_types
from numpy.core.overrides import integer
from numpy.core import empty, arange
from pylab import savefig

integer_types = integer_types + (integer,)


def fftfreq(n, d=1.0):
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val


def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [exp(-2j * pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]


samplerate, data = wavfile.read("output1.wav")

data = data[:, 0]
samples = data.shape[0]

datafft = fft(data)
# Get the absolute value of real and complex component:
fftabs = abs(datafft)
freqs = fftfreq(samples, 1 / samplerate)
plt.xlim([10, samplerate / 2])
plt.xscale('log')
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size / 2)])
plt.show()

savefig('outputfft.png', bbox_inches='tight')
