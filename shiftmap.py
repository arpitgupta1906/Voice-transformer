import matplotlib.pyplot as plt
import numpy as np
#from findpeaks import findpeaks
import pyaudio
import wave
from pylab import *
from scipy.io import wavfile

def plot(data):
    plt.plot(data, color ='steelblue')

    #LINEAR SHIFT OF FREQUENCIES

#in sbse pehle peak extraction krna hai

   
"""FILENAME1 = "input1.wav"
FILENAME2 = "output1.wav"
FILENAME3 = "third1.wav"
rate1, wav_data1 = wavfile.read(FILENAME1)
rate2, wav_data2 = wavfile.read(FILENAME2)  # isko dekhna hai
rate3, wav_data3 = wavfile.read(FILENAME3)"""

 # shift in by t*8000/length (wave ) frequencies (right shift)
# ye chutiyapa hai

def shiftleft(wav_data1):
    """
    shift in by t*8000/length (wave ) frequencies (right shift)
    this takes a parameter wav_data1 which is already definesd earlier
    """
    t = 300
    Y = np.fft.fft(wav_data1)
    L = len(Y)/2
    X = np.zeros(2*L)
    X =X.astype(dtype=np.complex)
    X[0:L-t] = Y[t:L]
    X[t+L:2*L] = Y[L:2*L-t]
    y1 = np.fft.ifft(X) #inverse transform
    return y1



#shift in by t*8000/length (wave ) frequencies (right shift)
def shiftright(wav_data1):
    """
    #shift in by t*8000/length (wave ) frequencies (right shift)
     this takes a parameter wav_data1 which is already definesd earlier in stratiing fuctions read the file

    """
    t = 300
    Y = np.fft.fft(wav_data1)
    L = len(Y)/2
    X = np.zeros(2*L)
    X =X.astype(dtype=np.complex)
    X[t:L] = Y[0:L-t]
    X[L:2*L-t] = Y[L+t:2*L]
    y1 = np.fft.ifft(X) #inverse transform
    return y1

#So consider the range frequencies of a source voice: (A;B) and a target voice's range of frequencies: (a; b). The desire is to shift frequencies: A --> a and B --> b.

#IDENTIFY SHIFT LENGTHS BETWEEN PEAKS 

#peak already been selected from both audio samples
#they are defined as peaks1 and peaks2

def shiftlengthbetweenpeaks(wav_data1,peaks1,peaks2):
    """
    #identfy shift lenghts between peaks 
    #peak already been selected from both audio samples
    #they are defined as peaks1 and peaks2 in function mapping regression arpit
    """
    L= len(wav_data1)
    n=len(peaks1)
    range1 = np.zeros(n+2)
    range2 = np.zeros(n+2)
    range1[n+1] = L-peaks1[n-1]
    range2[n+1] = L-peaks2[n-1]

    for i in range(0,n):
        range1[i+1] = peaks1[i] - range1[i]
        range2[i+1] = peaks2[i] - range2[i]
    # print range1
    # print range2

#find the scale to shrink or strech
    scale = range1/range2
    #print scale
    range1 = range1.astype(np.int16)
    range2 = range2.astype(np.int16)

    return range1,range2






#shift only the range of peaks
#this only shifts for the single sided FFT
#this data will need to be made symmetric afterwards

def shiftonlyrangeofpeaks(peaks1,peaks2,wav_data3,):
    """
    #shift only the range of peaks
    #this only shifts for the single sided FFT
    #this data will need to be made symmetric afterwards

    """
    n = len(peaks1)
    X3 = np.fft.fft(wav_data3)
    L = len(X3)/2
    new = np.zeros(L)
    new = new.astype(dtype = np.complex)
    if peaks1[0]<peaks2[0]:
        new[0:peaks1[0]] = X3[0:peaks1[0]]
    else:
        new[0:peaks2[0]] = X3[0:peaks2[0]]

    if peaks1[1] < peaks2[1]:
        new[peaks2[1]:L] = X3[peaks2[1]:L]
    else:
        new[peaks1[1]:L] = X3[peaks1[1]:L]
    k = 0
    for i in range(peaks2[0],peaks2[1]):
        index = peaks1[0] + floor(scale[2]*k)
        k+=1
        index = int(index)
        new[i] = X3[index]
    return new



#run this if you wish to duplicates with zeros
def replaceduplicatewithzero(peaks1,peaks2,new):
    """
    #run this if you wish to duplicates with zeros

    """
    for j in range(peaks2[0],peaks2[1]):
         if new[j] == new[j+1]:
            new[j] = 0.0

    return new        




