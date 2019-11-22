# from scipy.fftpack import fft,fftfreq
from cmath import exp, pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numpy.compat import integer_types
# from numpy.core.overrides import integer
from numpy.core import empty, arange
from pylab import *

# integer_types = integer_types + (integer,)


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


# samplerate, data = wavfile.read("source.wav")

# data = data[:, 0]
# samples = data.shape[0]

# datafft = fft(data)
# # Get the absolute value of real and complex component:
# fftabs = abs(datafft)
# freqs = fftfreq(samples, 1 / samplerate)
# plt.xlim([10, samplerate / 2])
# plt.xscale('log')
# plt.grid(True)
# plt.xlabel('Frequency (Hz)')
# plt.plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size / 2)])
# plt.show()

# # savefig('outputfft.png', bbox_inches='tight')


def plot(file_name):

    sampFreq, snd = wavfile.read(file_name)

    snd = snd / (2.**15)  # convert sound array to float pt. values

    s1 = snd[:, 0]  # left channel

    s2 = snd[:, 1]  # right channel

    n = len(s1)
    p = fft(s1)  # take the fourier transform of left channel

    m = len(s2)
    p2 = fft(s2)  # take the fourier transform of right channel

    nUniquePts = ceil((n+1)/2.0)
    p = p[0:nUniquePts]
    p = abs(p)

    mUniquePts = ceil((m+1)/2.0)
    p2 = p2[0:mUniquePts]
    p2 = abs(p2)

    p = p / float(n)
    p = p**2  
    if n % 2 > 0:  
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n)
    plt.plot(freqArray/1000, 10*log10(p), color='k')
    plt.xlabel('LeftChannel_Frequency (kHz)')
    plt.ylabel('LeftChannel_Power (dB)')
    plt.show()
    p2 = p2 / float(m) 
    p2 = p2**2  




# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
    if m % 2 > 0: # we've got odd number of points fft
         p2[1:len(p2)] = p2[1:len(p2)] * 2
    else:
         p2[1:len(p2) -1] = p2[1:len(p2) - 1] * 2 # we've got even number of points fft

    freqArray2 = arange(0, mUniquePts, 1.0) * (sampFreq / m);
    plt.plot(freqArray2/1000, 10*log10(p2), color='k')
    plt.xlabel('RightChannel_Frequency (kHz)')
    plt.ylabel('RightChannel_Power (dB)')
    plt.show()

plot("source.wav")
