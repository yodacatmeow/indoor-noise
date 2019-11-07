"""
Descriptions:
    This code converts audio clips in "SNU-B36-50" to "SNU-B36-50-mini"
    - Down sampling: 44.1 kHz -> 16 kHz
    - Sample length: ~ 5 sec  -> 975 ms
    - file format:   .m4a     -> .wav
"""

# Python modules
import numpy as np
import pandas as pd
import librosa
import os
import matplotlib.pyplot as plt

# fs_new; len(resampled audio); new file format
fs_new = 16000
len_new_s = 0.975   # Unit in sec
format_new = ".wav"

# Input audio path (folder)
audio_dir = 'audio_in'
# Ouput audi path (folder)
audio_dir_out = 'audio'

# Metadata
metadata = 'metadata-snu-b36-50.csv'

# data frame
meta_df = pd.read_csv(metadata)
meta_len = len(meta_df)

# Combine path
def combinePath(audio_dir, track_id):
    return os.path.join(audio_dir, track_id)

# Main
if __name__ == "__main__":

    # For every row in the metadata
    for i, row in meta_df.iterrows():
        # Audio path
        audio_path = combinePath(audio_dir, row['track-id'])
        print("processing:", audio_path)
        # Load an audio
        s_n, fs = librosa.load(audio_path, sr=None, mono=True)
        # Cut from "event-start" to "event-start" + 0.975 sec
        s_n = s_n[int(fs * row['event-start-s']): int(fs * (0.975 + row['event-start-s']))]
        # Down sampling
        s_n = librosa.resample(s_n, fs, fs_new)
        # Save the down sampled signal as .wav format
        out_audio_name = row['track-id'][:row['track-id'].find(".m4a")] + format_new; #print(out_audio_name)
        out_audio_path = combinePath(audio_dir_out, out_audio_name)
        librosa.output.write_wav(out_audio_path, s_n, fs_new)