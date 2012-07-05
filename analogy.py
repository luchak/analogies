#!/usr/bin/env python

import os
import wave

import matplotlib.pyplot
import numpy
import scipy.spatial

SAMPLE_RATE = 44100
SAMPLE_BYTES = 2
CHANNELS = 1

DATA_DIR = os.path.expanduser('~/Dropbox/Research/analogies/')

def ReadWav(filename):
  wav = None
  try:
    wav = wave.open(filename, 'r')
    channels = wav.getnchannels()
    sample_bytes = wav.getsampwidth()

    wav_data = wav.readframes(wav.getnframes())

    if (sample_bytes == 2):
      wav_array = numpy.fromstring(wav_data, dtype=numpy.int16)
    elif (sample_bytes == 1):
      wav_array = numpy.fromstring(wav_data, dtype=numpy.int8)
    else:
      raise ValueError('Sample depth of %d bytes not supported' % sample_bytes)

    float_array = numpy.zeros(wav_array.shape[0] / channels)
    for i in range(channels):
      float_array += wav_array[i::channels]
    float_array /= max(abs(float_array))
  finally:
    if wav:
      wav.close()

  return float_array

def WriteWav(filename, data, channels=1, sample_rate=44100):
  wav = None
  try:
    wav = wave.open(filename, 'w')
    wav.setnchannels(channels)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.setnframes(len(data) / channels)

    norm_data = (data * 32767 / (max(abs(data)))).astype(numpy.int16)

    wav.writeframes(norm_data.tostring())
  finally:
    if wav:
      wav.close()

def STFT(data):
  pxx, freqs, bins, im = matplotlib.pyplot.specgram(data, Fs=SAMPLE_RATE, NFFT=256, noverlap=128)
  #pxx = numpy.log(pxx)
  return pxx

def MakeFeatures(stfts):
  data = numpy.hstack(stfts)
  mean = numpy.mean(data, axis=1)
  u, s, v = numpy.linalg.svd(data, full_matrices=False)
  basis = u[:,:2]
  projected = []
  for stft in stfts:
    stft -= mean[:, numpy.newaxis]
    stft_proj = numpy.dot(basis.T, stft)
    print numpy.linalg.norm(numpy.dot(basis, stft_proj)) / numpy.linalg.norm(stft)
    projected.append(stft_proj)
  return projected

a = ReadWav(DATA_DIR + 'a.wav')
ap = ReadWav(DATA_DIR + 'ap.wav')
b = ReadWav(DATA_DIR + 'b.wav')

a_fft = STFT(a)
ap_fft = STFT(ap)
b_fft = STFT(b)

features = MakeFeatures([a_fft, ap_fft, b_fft])

#ann = scipy.spatial.cKDTree()

