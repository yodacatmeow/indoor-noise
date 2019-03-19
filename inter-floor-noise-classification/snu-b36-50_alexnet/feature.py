"""
Descriptions
    # This code converts a given audio file to a log-scaled Mel-spectrogram with size of 227 x 227
    # Time length of the log-scaled mel-spectrogram is "time_len" [sec]
    # Since the input of AlexNet is 227 x 227 x 3, the code stacks 3 patches
"""

# Public python modules
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# Audio path
def get_audio_path(audio_dir, track_id):
    audio_format = '.m4a'
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + audio_format)


# Patch time limit (refer 'event_start' in the metadata)
def patch_lim_meta(s_n, fs, event_start, time_len, offset = 0.05):
    index = [int(event_start - offset*fs), int(event_start + fs * (time_len - offset))]
    # - If the "index_end" > len(s_n)
    if index[1] > len(s_n):
        index = [int(len(s_n) - fs*time_len), len(s_n)]
        print("index[1] exceeds len(s_n)")
    else:
        pass
    return index


# Log-scaled Mel-spectrogram; returns [Mel-spec, Mel-spec, Mel-spec]
# Generate Mel-filterbank and multiply the filterbank with a spectrogram
def melspec2(s_n, fs, n_fft, win_length, hop_length, n_mels, fmax, power):
    n_width = n_mels
    #- Spectrogram; dtype = complex64
    S = librosa.core.stft(y=s_n, win_length=win_length, n_fft=n_fft, hop_length=hop_length)
    #- To power
    S = np.abs(S)**power
    #- Mel-filterbank
    fb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=fmax, norm=None)
    #- To Mel-spectrogram
    S = np.matmul(fb, S)
    #- To dB
    S = librosa.logamplitude(S, ref_power=np.max)
    #- Prepare return
    P = np.empty((n_width, n_mels, 3), dtype=np.float32)
    P[:, :, 0] = S
    P[:, :, 1] = S
    P[:, :, 2] = S
    return P


# Feature generation
def feature(tid, event_start=0):
    try:
        # Size of feature & offset [sec]
        n_mels = 227; n_width = n_mels; offset = 0.05
        # Time duration; window_length; FFT_point; hop_length
        time_len = 3; win_length = 583; n_fft = 2048; hop_length = 583
        # File path
        filepath = get_audio_path('audio', tid)
        # Read audio files
        s_n, fs = librosa.load(filepath, sr=None, mono=True)  # s_n = signal, fs = sampling freq.
        # If audio input is shorter than "time_len", then pads zeros
        if len(s_n) < time_len * fs:
            s_n = np.pad(s_n, (0, time_len * fs - len(s_n)), 'constant')
            print("sample is shorter than time_len [tid]:", tid)
        # Patch time limit (when mode = refer metadata)
        patchlim = patch_lim_meta(s_n, fs, event_start, time_len, offset)
        # Cut 's_n'
        s_n = s_n[patchlim[0]:patchlim[1]]
        # Patch
        patch = melspec2(s_n=s_n, fs=fs, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, fmax=int(fs/2), power=2)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        return False, 0
    return True, patch


if __name__ == "__main__":
    # Test0; This will returns: "dataset/train/054/054151.m4a"
    #print(get_audio_path('audio', 54151))

    # Test 1; e.g. feature(10049) draws "dataset/train/010/010049.m4a"
    result, feature = feature(4000, 71904); print(feature.shape)

    # Test 2; Draw a spectrum img; You need to return "patch_xx" instead of "rgb_patch"
    plt.figure(figsize=(8, 6)), librosa.display.specshow(feature[:,:,0], x_axis='time'), plt.colorbar(), plt.clim(np.min(feature), np.max(feature)), plt.show()

    # Test 3; BGR mode
    #plt.imshow(feature[:,:,0]); plt.show()


