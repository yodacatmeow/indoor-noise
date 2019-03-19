# Public python modules #
import numpy as np
import pandas as pd
import pickle
import feature
from os import path

class dict():
    def __init__(self, metadata_path, label_column_name, split):

        self.metadata_path = metadata_path
        self.label_column_name = label_column_name
        self.split = split
        self.label_dict = {}
        # Run "generate_data()" automatically
        self.get_label_dict()

    def get_label_dict(self):
        # meta-data
        meta_df = pd.read_csv(self.metadata_path)
        # Rows whose 'split' == self.split in the metadata
        meta_df = meta_df[meta_df['split'] == self.split]
        # Categories in "meta_df"
        #self.label_dict = {k: v for v, k in enumerate(sorted(set(meta_df[self.label_column_name].values)))}    # mode0: e.g. label: '0'
        self.label_dict = {v: k for v, k in enumerate(sorted(set(meta_df[self.label_column_name].values)))}     # mode1: e.g. 0: 'label'
        #print(self.label_dict)

class generate():
    def __init__(self, metadata_path, data_path, is_traindata, batch_size, label_column_name, split):
        self.batch_size = batch_size
        self.token_stream = []
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.is_traindata = is_traindata
        self.label_column_name = label_column_name
        self.split = split
        self.label_dict = {}
        # Run "generate_data()" automatically
        self.generate_data()

    def generate_data(self):
        # meta-data
        meta_df = pd.read_csv(self.metadata_path)
        # Rows whose 'split' == self.split in the metadata
        if self.is_traindata:
            meta_df = meta_df[(meta_df['split'] != self.split) & (meta_df['split'] != 'test') & (meta_df['split'] != 'nu')]
            print('Generating a training dataset')
        else:
            meta_df = meta_df[meta_df['split'] == self.split]
            print('Generating a validation dataset')
        # Categories in "meta_df"
        self.label_dict = {k: v for v, k in enumerate(sorted(set(meta_df[self.label_column_name].values)))}
        #print(self.label_dict)

        # Append
        tid_append = []                                             # Audio track ID
        class_append = []                                           # Class
        patch_append = []                                           # Patch

        #- Loop
        for i, row in meta_df.iterrows():
            tid = row['track_id']
            label = row[self.label_column_name]
            event_start = row['event_start']
            #-- Extract a patch
            result, patch = feature.feature(tid, event_start)
            #-- Append
            if result:
                tid_append.append(tid)
                class_append.append(self.label_dict.get(label))
                patch_append.append(patch)
                print('successfully extracted patch : {}'.format(tid))

        # Write appended array into data frame
        df = pd.DataFrame()
        df['track_id'] = tid_append
        df['category'] = class_append
        df['patch'] = patch_append

        # Shuffle rows (for better training)
        df = df.iloc[np.random.permutation(len(df))]

        self.data_frame = df
        self.num_class = len(self.label_dict)
        #print(self.num_class)

        # Save "data_frame" as .p (pickle)
        pickle.dump(self.data_frame, open(self.data_path, "wb"))

        # If you want to see the structure of "self.data_frame" (* It is not recommended),
        # print(self.data_frame)

if __name__ == "__main__":
    # A Simple module test
    metadata_path = 'dataset/metadata_box_case0.csv'
    traindata_path = 'dataset/train.p'
    validdata_path = 'dataset/valid.p'
    #- Generate training and validation set
    print("Generate a training set")
    test = generate(metadata_path, traindata_path, 10, 'category', split='training')
    print("Generate a validation set")
    test = generate(metadata_path, traindata_path, 10, 'category', split='validation')
