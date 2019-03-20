# Public python modules #
import numpy as np
import pandas as pd
import pickle
import feature
from os import path

# If categories of test data = categories of the training data
class load():
    def __init__(self, data_path, batch_size):
        self.pointer = 0
        self.dataframe = pickle.load(open(data_path,"rb"))
        self.batch_size = batch_size
        self.n_batch = int(len(self.dataframe) / self.batch_size)       # The number of batches
        self.n_class = len(set(self.dataframe['category'].values))      # The number of classes

    # Batch
    def batch(self, dataframe):
        x_data = []
        y_data = []
        # get patches from the saved data (in here, "/dataset/train.p" OR "/dataset/valid.p") and append
        for i, row in dataframe.iterrows():
            # Select dataframe[row, 'patch']
            patch = row['patch']
            # Append "patch" to "x_data"
            x_data.append(np.float32(patch))

            # One-hot encoding
            cl = row['category']
            y = np.zeros(self.n_class)
            y[cl] = 1
            y_data.append(y)
        return x_data, y_data
        #print("x:", x_data)
        #print("y:", y_data)

    # Mini-batch (via batch(self, dataframe) )
    def next_batch(self):
        start_pos = self.pointer * self.batch_size
        batch_df = self.dataframe.iloc[start_pos:start_pos + self.batch_size]
        self.pointer = (self.pointer + 1) % self.n_batch      # Move pointer for the next mini-batch
        return self.batch(batch_df)

    # Mini-batch (via batch(self, dataframe) )
    def next_batch(self):
        start_pos = self.pointer * self.batch_size
        batch_df = self.dataframe.iloc[start_pos:start_pos + self.batch_size]
        self.pointer = (self.pointer + 1) % self.n_batch      # Move pointer for the next mini-batch
        return self.batch(batch_df)

if __name__ == "__main__":
    # A simple test
    import gen_data


