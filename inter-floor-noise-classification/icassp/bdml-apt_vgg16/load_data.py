# Public python modules #
import numpy as np
import pandas as pd
import pickle
import feature
from os import path

# If categories of test data = categories of the training data
class load():
    def __init__(self, dataframe, batch_size, labels):                                      # "labels"
        self.pointer = 0
        self.dataframe = pickle.load(open(dataframe,"rb"))
        self.batch_size = batch_size
        self.labels = labels
        self.n_batch = int(len(self.dataframe) / self.batch_size)                           # The number of batches
        self.n_category = len(set(self.dataframe[labels]))                                  # The number of categories
        self.labels_dict = sorted(set(self.dataframe[labels]))                              # Dictionary of the categories

    # Global mean
    def global_mean(self):
        global_mean_value = self.dataframe['patch_mean'][0]
        return global_mean_value

    # Batch
    def batch(self, dataframe):
        x_data = []; y_data = []                                                            # Patch; label
        for i, row in dataframe.iterrows():                                                 # interators: i=index
            patch = row['patch']; x_data.append(np.float32(patch))                          # Patch -> "x_data"
            y = np.zeros(self.n_category); y[self.labels_dict.index(row[self.labels])]=1    # One-hot encoding
            y_data.append(y)                                                                # y -> y_data
        return x_data, y_data

    # Mini-batch
    def next_batch(self):
        # df of a mini-batch
        minibatch_df = self.dataframe.iloc[self.pointer * self.batch_size : self.pointer * self.batch_size + self.batch_size]
        # Update the pointer
        self.pointer = (self.pointer + 1) % self.n_batch                                   # % operator initializes the pointer to 0
        return self.batch(minibatch_df)

# Test:
if __name__ == "__main__":
    data = load(dataframe='train_type_3s_224_224_k1.p', batch_size=1, labels='type')

    for i in range(data.n_batch):
        data.next_batch()

    print(data.global_mean())

    print(data.n_category)