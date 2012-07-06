#!/usr/bin/env python

import os
import wave

import matplotlib.pyplot
import numpy
import scipy.spatial
import scipy.ndimage

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
  pxx, freqs, bins, im = matplotlib.pyplot.specgram(data, Fs=SAMPLE_RATE, NFFT=512, noverlap=256)
  pxx = numpy.log(pxx)
  return pxx

def MakeFeatures(stfts):
  data = numpy.hstack(stfts)
  mean = numpy.mean(data, axis=1)
  u, s, v = numpy.linalg.svd(data, full_matrices=False)
  basis = u[:,:10]
  projected = []
  for stft in stfts:
    stft -= mean[:, numpy.newaxis]
    stft_proj = numpy.dot(basis.T, stft)
    print numpy.linalg.norm(numpy.dot(basis, stft_proj)) / numpy.linalg.norm(stft)
    projected.append(stft_proj)
  return projected

def BinomialCoefs(size):
  coefs = numpy.array([1.0, 1.0])
  while len(coefs) < size:
    coefs = numpy.convolve(coefs, numpy.array([1.0, 1.0]))
  return coefs / coefs.sum()

def MakePyramid(data):
  pyramid = [data.copy()]
  coefs = BinomialCoefs(5)
  while data.shape[1] > len(coefs):
    data = scipy.ndimage.filters.convolve1d(data, coefs, axis=1)
    data = data[:,::2]
    pyramid.append(data.copy())
  pyramid.reverse()
  return pyramid
  

a = ReadWav(DATA_DIR + 'a.wav')
ap = ReadWav(DATA_DIR + 'ap.wav')
b = ReadWav(DATA_DIR + 'b.wav')

a_fft = STFT(a)
ap_fft = STFT(ap)
b_fft = STFT(b)

features = MakeFeatures([a_fft, ap_fft, b_fft])
pyramids = [MakePyramid(f) for f in features]
a_pyr = pyramids[0]
ap_pyr = pyramids[1]
b_pyr = pyramids[2]

result_pyramid = []

c3 = BinomialCoefs(3)
c5 = BinomialCoefs(5)

def CurrentLevelFeatureVector(a, ap, center, behind, ahead):
  vector = []
  for i in range(-behind, ahead + 1):
    vector.append(c5[i+2] * a[:,center+i])
    if (i < 0):
      vector.append(c5[i+2] * ap[:,center+i])
  return numpy.array(vector).flatten()

def UpperLevelFeatureVector(a, ap, center, behind, ahead):
  vector = []
  for i in range(-behind, ahead + 1):
    vector.append(c3[i+1] * a[:,center+i])
    vector.append(c3[i+1] * ap[:,center+i])
  return 1.6*numpy.array(vector).flatten()

index_pyr = []
for level in range(len(b_pyr)):
  indices = []
  for i in range(b_pyr[level].shape[1]):
    best = -1
    best_dist = 1e50

    look_behind = min(i, 2)
    look_ahead = min(b_pyr[level].shape[1] - i - 1, 2)
    if level > 0:
      i_upper = i/2
      look_behind_upper = min(i/2, 1)
      look_ahead_upper = min(b_pyr[level-1].shape[1] - i_upper - 1, 1)

    search_vec_current = CurrentLevelFeatureVector(b_pyr[level], b_pyr[level][:,indices], i, look_behind, look_ahead)
    if level > 0:
      search_vec_upper = UpperLevelFeatureVector(b_pyr[level-1], b_pyr[level-1][:,index_pyr[level-1]], i_upper, look_behind_upper, look_ahead_upper)

    for j in range(look_behind, a_pyr[level].shape[1] - look_ahead):
      diff_current = search_vec_current - CurrentLevelFeatureVector(a_pyr[level], ap_pyr[level], j, look_behind, look_ahead)
      dist = numpy.dot(diff_current, diff_current)
      if level > 0:
        j_upper = j/2
        diff_upper = search_vec_upper - UpperLevelFeatureVector(a_pyr[level-1], ap_pyr[level-1], j/2, look_behind_upper, look_ahead_upper)
        dist += numpy.dot(diff_upper, diff_upper)
      if dist < best_dist:
        best_dist = dist
        best = j

    indices.append(best)
  print indices
  index_pyr.append(indices)

triangle = numpy.concatenate((numpy.linspace(0.0, 1.0, 256), numpy.linspace(1.0, 0.0, 256)))
result = numpy.zeros((256 * (1 + b_fft.shape[1])))
final_idx = index_pyr[-1]
for i, idx, in enumerate(final_idx):
  result[i*256:i*256 + 512] += triangle * ap[indices[i]*256:indices[i]*256 + 512]

WriteWav('bp.wav', result)
