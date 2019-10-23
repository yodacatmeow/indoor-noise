"""
Descriptions:
    This code converts audio clips into a training data, a validation data and a test data
"""

# Python modules #
import numpy as np
import pandas as pd
import pickle
from feature import feature

def generate(metadata, output, tid, event_start, fold_conf, fold, labels, pp):
    # data frame
    meta_df = pd.read_csv(metadata)
    # Extract Rows of "meta_df" with "fold_conf" = "fold"
    meta_df = meta_df[meta_df[fold_conf].isin(fold)]

    # Track id, label, TF-patch
    tid_append = []; label_append = []; patch_append = []; patch_mean = 0.0; cnt = 0

    # Extract TF-patches (features) and append
    for i, row in meta_df.iterrows():
        #print(row[tid], row[labels], row[event_start]) # TEST CMD
        flag, patch = feature(tid=row[tid], time_len = pp['time_len'], n_fft = pp['n_fft'], win_size = pp['win_size'], hop_size = pp['hop_size'], fmax = pp['fmax'], adj_lim = pp['adj_lim'], width = pp['width'], event_start=row[event_start])
        if flag: tid_append.append(row[tid]); label_append.append(row[labels]); patch_append.append(patch); patch_mean += np.mean(patch); cnt = cnt + 1;
    # Global mean of the patches
    patch_mean = patch_mean / cnt
    # Write the appended arrays into a dataframe
    df = pd.DataFrame()
    df['tid'] = tid_append; df[labels] = label_append; df['patch'] = patch_append; df['patch_mean'] = patch_mean

    # Shuffle rows (for better training)
    df = df.iloc[np.random.permutation(len(df))]

    # Save "df" as .p(Pickle)
    pickle.dump(df, open(output, "wb"))

    # Print
    print(str(output),"is generated!")


# Test
if __name__ == "__main__":
    # Feature parameters
    pp_f = {'time_len':3.0, 'n_fft':2048, 'win_size':591, 'hop_size':591, 'adj_lim':0.0, 'width':224,'fmax': int(44100/2)}
    #pp_f = {'time_len': 1.5, 'n_fft': 2048, 'win_size': 296, 'hop_size': 296, 'adj_lim': 0.0, 'width': 224, 'fmax': int(44100 / 8)}
    # Names
    pp_n = {'output_tr':'train_floor_3s_224_224',            # 'type' <-> 'position'
            'output_v':'valid_floor_3s_224_224',             # 'type' <-> 'position'
            'output_t':'test_floor_3s_224_224',              # 'type' <-> 'position'
            'fold_conf':'fold_conf_3',                       # 'fold_conf_x' (select one in the metadata)
            'labels': 'upper_lower'}                         # 'type' <-> 'position'

    # training data
    for fold in range(1, 6):
        fold_list = ['1', '2', '3', '4', '5']
        fold_list.remove(str(fold))
        output = pp_n['output_tr'] + '_k' + str(fold) + '.p'
        generate(metadata='metadata.csv', output=output, tid='track_id', event_start='event_start_s', fold_conf=pp_n['fold_conf'], fold=fold_list, labels=pp_n['labels'], pp=pp_f)

    # Validation data
    for fold in range(1, 6):
        fold_list = [str(fold)]
        output = pp_n['output_v'] + '_k' + str(fold) + '.p'
        generate(metadata='metadata.csv', output=output, tid = 'track_id', event_start='event_start_s', fold_conf=pp_n['fold_conf'], fold=fold_list, labels=pp_n['labels'], pp=pp_f)

    # Test data
    fold_list = ['test']
    output = pp_n['output_t'] + '.p'
    generate(metadata='metadata.csv', output=output, tid='track_id', event_start='event_start_s', fold_conf=pp_n['fold_conf'], fold=fold_list, labels=pp_n['labels'], pp=pp_f)