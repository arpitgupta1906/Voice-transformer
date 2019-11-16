import numpy as np

#this only shifts for the single sided fft
#the data will need to be made symmetric afterwards

def shiftallpeaks(peaks1,peaks2,wav_data3):
    """ Shift all peaks paramaters(peaks1,peaks2,wav_data3)
    """

    n=len(peaks1)
    x3=np.fft.fft(wav_data3)
    new=np.zeros(len(x3)/2)
    new=new.astype(dtype=np.complex)

    Peaks1=np.zeros(n+2)
    Peaks1[1:n+1]=peaks1
    Peaks1[n+1]=len(x3)/2
    Peaks1=Peaks1.astype(np.int16)

    Peaks2=np.zeros(n+2)
    Peaks2[1:n+1]=peaks2
    Peaks2[n+1]=len(x3)/2

    Peaks2=Peaks2.astype(np.int16)

    for j in range(0,n+1):
        k=0

        for i in range(Peaks2[j],Peaks2[j+1]):
            index=Peaks1[j]+floor(scale[j+1]*k)
            k+=1
            index=int(index)
            new[i]=x3[index]

        #run this if you wish to replace duplicates with zeros

    for j in range(len(new)-1):
        if new[j]==new[j+1]:
            new[j]=0.0

    return new