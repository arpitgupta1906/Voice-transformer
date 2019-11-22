import cmath, sys
from scipy.io import wavfile
import matplotlib.pyplot as plt


if sys.version_info.major == 2:
	range = xrange


# 
# Computes the discrete Fourier transform (DFT) or inverse transform of the given complex vector, returning the result as a new vector.
# The vector can have any length. This is a wrapper function. The inverse transform does not perform scaling, so it is not a true inverse.
# 
def transform(vector, inverse):
	n = len(vector)
	if n == 0:
		return []
	elif n & (n - 1) == 0:  # Is power of 2
		return transform_radix2(vector, inverse)
	else:  # More complicated algorithm for arbitrary sizes
		return transform_bluestein(vector, inverse)


# 
# Computes the discrete Fourier transform (DFT) of the given complex vector, returning the result as a new vector.
# The vector's length must be a power of 2. Uses the Cooley-Tukey decimation-in-time radix-2 algorithm.
# 
def transform_radix2(vector, inverse):
	# Returns the integer whose value is the reverse of the lowest 'bits' bits of the integer 'x'.
	def reverse(x, bits):
		y = 0
		for i in range(bits):
			y = (y << 1) | (x & 1)
			x >>= 1
		return y
	
	# Initialization
	n = len(vector)
	levels = n.bit_length() - 1
	if 2**levels != n:
		raise ValueError("Length is not a power of 2")
	# Now, levels = log2(n)
	coef = (2 if inverse else -2) * cmath.pi / n
	exptable = [cmath.rect(1, i * coef) for i in range(n // 2)]
	vector = [vector[reverse(i, levels)] for i in range(n)]  # Copy with bit-reversed permutation
	
	# Radix-2 decimation-in-time FFT
	size = 2
	while size <= n:
		halfsize = size // 2
		tablestep = n // size
		for i in range(0, n, size):
			k = 0
			for j in range(i, i + halfsize):
				temp = vector[j + halfsize] * exptable[k]
				vector[j + halfsize] = vector[j] - temp
				vector[j] += temp
				k += tablestep
		size *= 2
	return vector


# 
# Computes the discrete Fourier transform (DFT) of the given complex vector, returning the result as a new vector.
# The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
# Uses Bluestein's chirp z-transform algorithm.
# 
def transform_bluestein(vector, inverse):
	# Find a power-of-2 convolution length m such that m >= n * 2 + 1
	n = len(vector)
	if n == 0:
		return []
	m = 2**((n * 2).bit_length())
	
	coef = (1 if inverse else -1) * cmath.pi / n
	exptable = [cmath.rect(1, (i * i % (n * 2)) * coef) for i in range(n)]  # Trigonometric table
	a = [(x * y) for (x, y) in zip(vector, exptable)] + [0] * (m - n)  # Temporary vectors and preprocessing
	b = exptable[ : n] + [0] * (m - (n * 2 - 1)) + exptable[ : 0 : -1]
	b = [x.conjugate() for x in b]
	c = convolve(a, b, False)[ : n]  # Convolution
	return [(x * y) for (x, y) in zip(c, exptable)]  # Postprocessing


# 
# Computes the circular convolution of the given real or complex vectors, returning the result as a new vector. Each vector's length must be the same.
# realoutput=True: Extract the real part of the convolution, so that the output is a list of floats. This is useful if both inputs are real.
# realoutput=False: The output is always a list of complex numbers (even if both inputs are real).
# 
def convolve(x, y, realoutput=True):
	assert len(x) == len(y)
	n = len(x)
	x = transform(x, False)
	y = transform(y, False)
	for i in range(n):
		x[i] *= y[i]
	x = transform(x, True)
	
	# Scaling (because this FFT implementation omits it) and postprocessing
	if realoutput:
		return [(val.real / n) for val in x]
	else:
		return [(val / n) for val in x]



# FILENAME1 = "source.wav"
# FILENAME2 = "map.wav"
# FILENAME3 = "target.wav"
# rate1, wav_data1 = wavfile.read(FILENAME1)
# rate2, wav_data2 = wavfile.read(FILENAME2)  # isko dekhna hai
# rate3, wav_data3 = wavfile.read(FILENAME3)

# x = transform(wav_data1, False)
# y = transform(x, True)

# plt.plot(x)
# plt.show()

# plt.plot(y)
# plt.show()