# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:27:53 2021

@author: srika
"""

import numpy as np
import math
import pylab as p
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

N=4802E2
N=int(N)
tmax=5
t=np.linspace(0,tmax,N)
y=np.ones((N))
nf=7
pi=np.pi
for i in range(0,nf):
    f=i*4
    y[int(i*N/nf):int((i+1)*N/nf)]=np.sin(2*pi*f*t[0:int(N/nf)])
    
plt.plot(t,y)

color_plot=['r','g','b','k','y','c','m']


plt.figure()

bin_size=(N/(nf*2))
freq_bin=np.arange(-bin_size,bin_size)*nf/tmax

freq_chart=np.zeros((nf,2))
'''
        without hanning window
'''
for i in range(0,nf):
    y1=y[int(i*N/nf):int((i+1)*N/nf)]
    y1_fft=np.fft.fft(y1)
    y1_fftshift=np.fft.fftshift(y1_fft)
    plt.plot(freq_bin,abs(y1_fftshift),color=color_plot[i])
    plt.xlim([-30,30])
    peaks=np.argmax(abs(y1_fftshift))
    freq_chart[i,0]=freq_bin[peaks]
'''
plt.figure()
t1=t[0:int(N/nf)]
plt.plot(t1,y1)
plt.axvline(x=0.5,color='r')
plt.axhline(y=0,color='r')
plt.savefig('Windowing problem', dpi=200)
'''
'''
            with hanning window
            Window=-M/2 to M/2
            Hanning=1/2*(1+cos(2*pi*n/M))
'''
plt.figure()
M=N/nf
window=np.linspace(-N/(nf*2),N/(nf*2),int(M))
Hanning=1/2*(1+np.cos(2*pi*window/M))
plt.plot(Hanning, color='k')
for i in range(0,nf):
    y1=y[int(i*N/nf):int((i+1)*N/nf)]
    y1_hanning=y1*Hanning
    y1_fft=np.fft.fft(y1_hanning)
    y1_fftshift=np.fft.fftshift(y1_fft)
    plt.plot(freq_bin,abs(y1_fftshift),color=color_plot[i])
    peaks=np.argmax(abs(y1_fftshift))
    freq_chart[i,1]=freq_bin[peaks] 
plt.xlim([-30,30])
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT absolute (no units)')
plt.savefig("Thisplot",dpi=200)
freq_chart[0,:]=0
stft_alike=np.ones((N))
for i in range(0,nf):
    stft_alike[int(i*N/nf):int((i+1)*N/nf)]=abs(freq_chart[i,1])
plt.figure()
plt.plot(t,y*25,color='r')
plt.plot(t,stft_alike,color='g',linewidth=4)
plt.legend(['Normalised signal','Frequency'])
plt.xlabel('Time(sec)')
plt.ylabel('Signal, Frequency (Hz)')
#plt.savefig('STFT_results',dpi=250)
y2=np.tile(y1_fftshift,7)


gridsize=(3,2)
fig=plt.figure(figsize=(12,8))
ax1=plt.subplot2grid(gridsize, (0,0), colspan=2, rowspan=2)
ax2=plt.subplot2grid(gridsize, (2,0))
ax3=plt.subplot2grid(gridsize, (2,1))
ax1.plot(t,y*25,color='r')
ax1.plot(t,stft_alike,color='g',linewidth=4)
ax1.legend(['Normalised signal','Frequency'])
ax1.set_xlabel('Time(sec)')
ax1.set_ylabel('Signal, Frequency (Hz)')
'''
ax2.plot(freq_bin,abs(y1_fftshift),color=color_plot[6])
ax2.set_xlim(-30,30)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('FFT absolute (no units)')
'''
import cv2
Thisplotimg=cv2.imread(r'Thisplot.png')
Thisplotimg=cv2.cvtColor(Thisplotimg, cv2.COLOR_BGR2RGB)
ax2.imshow(Thisplotimg)
ax2.axis('off')
ax3.plot(t,y)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Signal magnitude')
plt.tight_layout()
fig.savefig("STFT.pdf",dpi=250)
