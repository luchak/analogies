#!/usr/bin/env python

# extract raw (linear) features
# make pyramids
# nonlinear features per pyramid level

import math
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
  return pxx

def MakeFeatures(stfts):
  data = numpy.hstack(stfts)
  mean = numpy.mean(data, axis=1)
  u, s, v = numpy.linalg.svd(data, full_matrices=False)
  basis = u[:,:80]
  projected = []
  for stft in stfts:
    stft -= mean[:, numpy.newaxis]
    stft_proj = numpy.dot(basis.T, stft)
    print numpy.linalg.norm(numpy.dot(basis, stft_proj)) / numpy.linalg.norm(stft)
    projected.append(stft_proj)
  return projected

coef_cache = {}
def BinomialCoefs(size):
  global coef_cache
  if size in coef_cache:
    return coef_cache[size]
  coefs = numpy.array([1.0, 1.0])
  while len(coefs) < size:
    coefs = numpy.convolve(coefs, numpy.array([1.0, 1.0]))
  return coefs / coefs.sum()

def MelToHz(mel):
  return 700.0 * (math.exp(mel / 1127.0) - 1.0)

def HzToMel(hz):
  return 1127.0 * math.log(1.0 + hz / 700.0)

def HzToBin(fft_bands, freq):
  return int(math.floor(freq * fft_bands * 2.0 / float(SAMPLE_RATE)))

def MakeMelFilterbank(mel_bands, fft_bands):
  result = numpy.zeros((mel_bands, fft_bands + 1))
  low_mel = HzToMel(40.0)
  high_mel = HzToMel(16000.0)
  center_mels = numpy.linspace(low_mel, high_mel, mel_bands + 2)

  center_bins = [HzToBin(fft_bands, MelToHz(mel)) for mel in center_mels]

  for i in range(mel_bands):
    width_below = float(center_bins[i+1] - center_bins[i])
    width_above = float(center_bins[i+2] - center_bins[i+1])
    for j in range(center_bins[i], center_bins[i+1]+1):
      result[i,j] = (j - center_bins[i]) / width_below
    for j in  range(center_bins[i+1]+1, center_bins[i+2]):
      result[i,j] = (center_bins[i+2] - j) / width_above

  return result

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
  return 2.0*numpy.array(vector).flatten()

class Pyramid1D(object):
  def __init__(self, data):
    self.levels = [data.copy()]
    coefs = BinomialCoefs(5)
    while (self.levels[-1].shape[1] > len(coefs)):
      blur = scipy.ndimage.filters.convolve1d(self.levels[-1], coefs, axis=1)
      self.levels.append(blur[:,::2])
    self.levels.reverse()

  def Map(self, fn):
    """fn must transform a 2D array of column feature vectors to another 2D
    array of column feature vectors.
    """

    self.levels = [fn(level) for level in self.levels]

  def __len__(self):
    return len(self.levels)

  def __getitem__(self, key):
    return self.levels[key]

def MakeLogMelBands(data, linear_filter):
  return numpy.log(numpy.dot(linear_filter, data))

def Matrix1DFFT(data):
  pass


def SquaredDistance(a, b):
  diff = a - b
  return numpy.dot(diff, diff)

if __name__ == '__main__':
  a = ReadWav(DATA_DIR + 'a.wav')
  ap = ReadWav(DATA_DIR + 'ap.wav')
  b = ReadWav(DATA_DIR + 'b.wav')

  a_fft = STFT(a)
  ap_fft = STFT(ap)
  b_fft = STFT(b)

  a_pyr = Pyramid1D(a_fft)
  ap_pyr = Pyramid1D(ap_fft)
  b_pyr = Pyramid1D(b_fft)

  mel_coefs = MakeMelFilterbank(19, 256)
  a_pyr.Map(lambda x: MakeLogMelBands(x, mel_coefs))
  ap_pyr.Map(lambda x: MakeLogMelBands(x, mel_coefs))
  b_pyr.Map(lambda x: MakeLogMelBands(x, mel_coefs))

  #features = MakeFeatures([a_fft, ap_fft, b_fft])
  #pyramids = [MakePyramid(f) for f in features]
  #a_pyr = pyramids[0]
  #ap_pyr = pyramids[1]
  #b_pyr = pyramids[2]

  result_pyramid = []

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
        dist = SquaredDistance(search_vec_current, CurrentLevelFeatureVector(a_pyr[level], ap_pyr[level], j, look_behind, look_ahead))
        if level > 0:
          dist += SquaredDistance(search_vec_upper, UpperLevelFeatureVector(a_pyr[level-1], ap_pyr[level-1], j/2, look_behind_upper, look_ahead_upper))
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
