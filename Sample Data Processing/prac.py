import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.wav"

# This is a sample data preprocessing file done with sample audio to understand the working & concept of sound

#WAVEFORM
signal, sr =  librosa.load(file, sr=22050) # sr * T -> 22050*30 ( T = 30 secs)
# signal is 1D array that will contain sr*T values
librosa.display.waveshow(signal,sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.savefig('waveform.png')
plt.show()




#to convert from time domain to frequency domain we need to perfom fast fourier transfrom (fft)

#FFT -> SPECTRUM
#perform fourier transform
fft = np.fft.fft(signal) #fft is the array of complex no.s

#calculate absolute values on complex nos to get magnitude
magnitude = np.abs(fft) #magnitude is array of absolute values

#spectrum(magnitude) tells you how much of each frequency is present in the signal.
#frequency tells you which specific frequency each number in the spectrum is related to.

freq = np.linspace(0,sr,len(magnitude)) #when plotted forms a symmetric graph and we consider only left side hence left_freq
left_freq = freq[:int(len(freq)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

plt.plot(left_freq,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.savefig('spectrum.png')
plt.show()
#this is a snapshot of the sound since we lost the information about the time 

#STFT (Short Time Fourier Transform) -> Spectrogram
#computes several Fourier transforms at different intervals and in doing so it preserves information about time

n_fft = 2048 # no of samples in each window/frame

#in stft we slide through an interval and at each interval calculate 

#calculate duration of hop length and window in seconds
hop_length = 512 # in num. of samples

# calculate duration hop length and window in seconds
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft = n_fft)

spectrogram = np.abs(stft)

librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")
plt.savefig('spectogram.png')
plt.show()

# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram, sr=sr,
hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.savefig('spectogram_log.png')
plt.show()

# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
# display MFCCs
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")
plt.savefig('mfcc.png')
plt.show()