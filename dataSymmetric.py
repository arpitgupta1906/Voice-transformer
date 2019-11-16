import matplotlib.pyplot as plt
import numpy as np


def plot(data):
    plt.plot(data, color='steelblue')
def dataSymmetric(new):
    L=len(new)
    x3 = np.zeros(2*L)
    x3 = x3.astype(dtype=np.complex)
    x3[0] = new[0]
    for i in range(1,L):
        x3[i]=new[i]
        x3[2*L-i]=complex(real(new[i]),-imag(new[i]))
    plot(x3)
