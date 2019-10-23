"""
Descriptions:
    This code converts a given audio file to a log-scaled Mel-spectrogram with size of 224 x 224.
    "time_len" [sec] sets the time length of the log-scaled Mel-spectrogram.
    VGG16 is designed for image recognition and it has three input channels.
    The log-scaled Mel-spectrogram is provided to all input channels of VGG16.

Option:
    - standardization
"""

# Public python modules
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn import preprocessing

# Audio path
def get_audio_path(audio_dir, track_id):

    return os.path.join(audio_dir, track_id)


# Selecting time range
def time_lim(s_n, fs, time_len, event_start, adj_lim = 0.05):
    index = [int(event_start * fs - adj_lim * fs), int( fs * (event_start + time_len - adj_lim))]
    # If "index[end]" > len(s_n):
    if index[1] > len(s_n):
        index = [int(len(s_n) - fs * time_len), len(s_n)]
        #print("index[1] exceeds len(s_n)")
    else:
        pass
    return index


# Log-scaled Mel-spectrogram
def melspec2(s_n, fs, n_fft, win_size, hop_size, n_mels, fmax, power):
    n_width = n_mels
    # Spectrogram; dtype = complex64
    S = librosa.core.stft(y=s_n, win_length=win_size, n_fft=n_fft, hop_length=hop_size)
    # To power
    S = np.abs(S)**power
    # Mel-filterbank
    fb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=fmax, norm=None)
    # To Mel-spectrogram
    S = np.matmul(fb, S)
    # To dB
    S = librosa.logamplitude(S, ref_power=np.max)
    # Prepare return
    P = np.empty((n_width, n_mels, 3), dtype=np.float32)
    P[:, :, 0] = S; P[:, :, 1] = S; P[:, :, 2] = S
    return P


# Extract a feature from an audio clip (default paramemters)
def feature(tid, time_len = 3.0, n_fft = 2048, win_size = 591, hop_size = 591, fmax= int(44100/2), adj_lim = 0.0, width = 224, event_start=0):
    try:
        # Parameters
        # time_len: Time length of the window
        # n_fft: n points FFT
        # win_size: widow size
        # hop_size: Hop size
        # adj_lim: Adjust position of "event_start" (move minus "adj_lim" second)
        # width: Width of the feature
        # height: Height of the feature
        height = width

        filepath = get_audio_path('../1_dataset/indoor-noise', tid)
        s_n, fs = librosa.load(filepath, sr=None, mono=True)    # s_n = signal; fs = sampling freq.

        # If audio clip is very short:
        if len(s_n) < time_len * fs: s_n = np.pad(s_n, (0, int(time_len * fs) - len(s_n)), 'constant')
        # Patch time limit (when mode = refer metadata)
        time_lim_indices = time_lim(s_n=s_n, fs=fs, time_len=time_len, event_start=event_start, adj_lim=adj_lim)
        # Cut 's_n'
        s_n = s_n[time_lim_indices[0]:time_lim_indices[1]]
        # Standardization
        #s_n = preprocessing.scale(s_n)
        # Patch
        patch = melspec2(s_n=s_n, fs=fs, n_fft=n_fft, win_size=win_size, hop_size=hop_size, n_mels=height, fmax=fmax, power=2)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        return False, 0
    return True, patch

# Test
if __name__ == "__main__":

    # Test 1: ex) feature(tid = 0) draws "audio/000/000000.m4a"
    result, feature = feature(tid = 'C-063040.m4a',event_start=0.7)  # using the default parameters

    # Test 2: Draw the feature; You need to return "patch_xx" instead of "rgb_patch"
    plt.figure(figsize=(8, 6)), librosa.display.specshow(feature[:,:,0], x_axis='time'), plt.colorbar(), plt.clim(np.min(feature), np.max(feature)), plt.show()