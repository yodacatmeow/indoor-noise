"""
Descriptions:
    This code converts audio clips into a training / validation data and a test data
"""

# Python modules
import numpy as np
import pandas as pd
import pickle
import os
from sklearn import preprocessing
import scipy
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
#print(librosa.__version__)

# Get an audio path
def get_audio_path(audio_dir, track_id):
    return os.path.join(audio_dir, track_id)

# Get an event start via HOS (high order statistics)
def get_event_start_hos_hop(signal, fs=44100, cut_init_s=0.5, win_len=3000, hop = 10, adj_s=0.03, fisher=True):
    # Cut initial parts
    signal  = signal[int(fs * cut_init_s):-1]
    # length of kurtosis result array
    gamma_len = len(signal) - win_len + 1
    # Init a kurtosis array
    gamma = np.zeros(gamma_len)
    # slide the window and calculate kurtosis
    for i in range(int(gamma_len/hop)):
        gamma[i*hop] = kurtosis(signal[i * hop : i * hop + win_len - 1], fisher = fisher)
    # Resize the kurtosis array
    gamma = np.resize(gamma, len(signal))
    # Localize the first onset
    onset = int(np.argmax(gamma) + cut_init_s * fs + adj_s * fs)
    return onset


# Log-Mel power spectrogram
def log_mel_spec_power(audio_path,
                       signal_len_s=3,
                       event_start_s=0,
                       n_fft=2048,
                       win_size=591,
                       hop_size=591,
                       adj_lim=0,
                       patch_size=224,
                       fmax=22050,
                       power=2
                       ):

    try:
        n_mel = patch_size
        # audio
        signal, fs = librosa.load(audio_path, sr=None)

        # If the signal has length shorter than the defined "sig_len_s"
        if len(signal) < signal_len_s * fs: signal = np.pad(signal, (0, int(signal_len_s * fs) - len(signal)), 'constant')
        # Cut the audio signal based on the metadata
        event_start_loc = get_event_start_hos_hop(signal)
        cutting_interval = [int(event_start_loc + adj_lim * fs), int(event_start_loc + signal_len_s * fs + adj_lim * fs)]

        # If "index[end]" > len(s_n)
        if cutting_interval[-1] > len(signal): cutting_interval = [int(len(signal) - fs * signal_len_s), len(signal)]

        signal = signal[cutting_interval[0]: cutting_interval[-1]]
        # Adjust amplitude to [-1,1]
        signal = librosa.util.normalize(signal)

        # Spectrogram; dtype= complex64
        S = librosa.core.stft(y=signal, win_length=win_size, n_fft=n_fft, hop_length=hop_size)
        # To power
        S = np.absolute(S) ** power
        # Mel-filterbank
        fb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mel, fmin=0, fmax=fmax, norm=None)
        # To Mel-spectrogram
        S = np.matmul(fb, S)
        # To dB
        S = librosa.logamplitude(S, ref_power=np.max)


    except Exception as e:
        print('{}: {}'.format(audio_path, repr(e)))
        return False, 0

    return True, S


def dataset(pp_d, pp_f, folds, output):
    # df of metadata
    metadata_df = pd.read_csv(pp_d['metadata'])
    # Select rows whose "fold_config" == "fold"
    metadata_df = metadata_df[metadata_df[pp_d['fold_config']].isin(folds)]

    # Track id, label, TF-patch
    tid_append = []
    label_append = []
    patch_append = []

    # Extract log-Mel Power spectrogram
    for i, row in metadata_df.iterrows():

        audio_path = get_audio_path(pp_d['audio_folder'], track_id=row['track-id'])
        # Print out the current audio path
        print(audio_path)
        # Convert the given audio signal to a patch
        flag, patch = log_mel_spec_power(audio_path=audio_path,
                                         signal_len_s=pp_f['signal_len_s'],
                                         event_start_s=row[pp_d['event_start_s_col']],
                                         n_fft=pp_f['n_fft'],
                                         win_size=pp_f['win_size'],
                                         hop_size=pp_f['hop_size'],
                                         adj_lim=pp_f['adj_lim'],
                                         patch_size=pp_f['patch_size'],
                                         fmax=pp_f['fmax'],
                                         power=pp_f['power']
                                         )

        # Stack an identical patch
        P = np.empty((pp_f['patch_size'], pp_f['patch_size'], 3), dtype=np.float32)
        P[:, :, 0] = patch
        P[:, :, 1] = patch
        P[:, :, 2] = patch

        if flag:
            # Append
            tid_append.append(row[pp_d['track_id_col']])
            label_append.append(row[pp_d['labels']])
            patch_append.append(P)


        # Pickle dataframe
        df = pd.DataFrame()
        df['tid'] = tid_append
        df[pp_d['labels']] = label_append
        df['patch'] = patch_append

        # Shuffle rows (for better training)
        df = df.iloc[np.random.permutation(len(df))]

        # Save "df" as .p(Pickle)
        pickle.dump(df, open(output, "wb"))

        # Test commands
        #plt.figure(figsize=(8, 6)), librosa.display.specshow(P[:, :, 0], x_axis='time'), plt.colorbar(), plt.clim(np.min(P), np.max(P)), plt.show()
        #scipy.misc.imsave('toImage/' + row['track-id'] + '.png', P[:, :, 0])


# Main
if __name__ == "__main__":

    # Feature parameters 0
    pp_f = {
        'signal_len_s': 0.152,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 30,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }

    # Feature parameters 1
    pp_f1 = {
        'signal_len_s': 0.501,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 99,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }

    # Feature parameters 2
    pp_f2 = {
        'signal_len_s': 1.0,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 197,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }

    # Feature parameters 3
    pp_f3 = {
        'signal_len_s': 1.5,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 296,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }

    # Feature parameters 4
    pp_f4 = {
        'signal_len_s': 2.0,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 394,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }

    # Feature parameters 5
    pp_f5 = {
        'signal_len_s': 3.0,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 591,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }


    # Feature parameters test
    pp_f_test = {
        'signal_len_s': 0.152,  # Unit: sec
        'n_fft': 8192,  # n-point FFT
        'win_size': 1024,  # Hanning window size; #of sample to be Fourier Transformed after zero padding; Unit: Sample;
        'hop_size': 30,  # Unit: sample
        'adj_lim': 0,  # Adjust event start; Unit: sec
        'patch_size': 224,  # The number of kernels
        'fmax': 2000, # fs/2; Unit: Hz
        'power': 2
    }


    # Dataset
    pp_d = {
        'audio_folder': '../1_dataset/indoor-noise',
        'metadata': 'metadata.csv',
        'output_tr': 'train_CS_type_all_test',  # Training data
        'output_v': 'valid_CS_type_all_test',  # Validation data
        'output_t': 'test_CS_type_all_test',  # Test data
        'track_id_col': 'track-id',
        'fold_config': 'CS_TEST_ALL',  # select one in the metadata; e.g. CS_test-A ...
        'labels': 'type',  # select one in the metadata; e.g. type, floor ...
        'event_start_s_col': 'event-start-s',
        'folds': ['1','2','3','4','5']
    }

    # Training set
    for fold in pp_d['folds']:
        folds = pp_d['folds'].copy()
        folds.remove(fold)
        output = pp_d['output_tr'] + '_k' + fold + '.p'
        dataset(pp_d=pp_d, pp_f=pp_f5, folds=folds, output=output)

        print(output, "generated!")

    # Validation set
    for fold in pp_d['folds']:
        folds = [fold]
        # Name of an output file
        output = pp_d['output_v'] + '_k' + fold + '.p'
        dataset(pp_d=pp_d, pp_f=pp_f5, folds=folds, output=output)

        print(output, "generated!")

    # Test set
    dataset(pp_d=pp_d, pp_f=pp_f5, folds=['test'], output=pp_d['output_t']+'.p')

