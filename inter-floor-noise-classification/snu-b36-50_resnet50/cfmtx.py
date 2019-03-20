"""
"cfmtx.py"

References
    # Plot a confusion matrix
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
# Public python modules#
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Custom python module
from gen_data import generate
from gen_data import dict

# If categories of a validation set is subset of categories of a training set
def cfmtx(label, prediction, dim, batch_size):
    c = np.zeros([dim,dim])     # Initialize c confusion matrix
    l = label                   # Label
    p = prediction              # Prediction
    for i in range(batch_size):
        c[l[i], p[i]] += 1
    return c

# If categories of test set is not a subset of categories of training set
def cfmtx2(label, prediction, shape):
    c = np.zeros([shape[0], shape[1]])              # Initialize confusion matrix
    l = label                                       # Label
    p = prediction                                  # Prediction
    c[l, p] += 1
    return c

# Merge k confusion matrices in .csv
def merge(folder, dim, keyword, n_data_per_category, normalize = True):
    dir = os.listdir(folder)
    filenames = []
    for file in dir:
        if file.find(keyword) == -1:
            pass
        else:
            filenames.append(os.path.join(folder,file))

    c = np.zeros([dim, dim])
    for file in filenames:
        # Read  csv file
        df = pd.read_csv(file, sep=',')
        # Drop the first col.
        df = df.drop(df.columns[0], axis=1)
        c = c + df
    c = np.array(c) # numpy array
    if normalize:
        c = c / n_data_per_category
    else:
        pass
    return c

# If categories of validation set is subset of categories of training set
def draw(file, normalize = True, ticks_ref=None):
    # Read  csv file
    df = pd.read_csv(file, sep=',')
    # Drop the first col.
    df = df.drop(df.columns[0], axis=1)
    cfmtx = np.array(df)  # numpy array
    # Fill
    plt.imshow(cfmtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # Ticks
    tick_marks = np.arange(len(ticks_ref))
    plt.xticks(tick_marks, ticks_ref, rotation=90)
    plt.yticks(tick_marks, ticks_ref)
    # Text
    fmt = '.2f' if normalize else 'd'
    thres = cfmtx.max() / 2.
    for i, j in itertools.product(range(cfmtx.shape[0]), range(cfmtx.shape[1])):
        #print(i, j)
        plt.text(j, i, int(cfmtx[i, j]), horizontalalignment='center', verticalalignment='center', color='white' if cfmtx[i, j] > thres else 'black')
    plt.tight_layout()
    # Label
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# If categories of test set is not a subset of categories of training set
def draw2(file, normalize = True, xticks_ref=None, yticks_ref=None):
    # Read  csv file
    df = pd.read_csv(file, sep=',')
    # Drop the first col.
    df = df.drop(df.columns[0], axis=1)
    cfmtx = np.array(df) # numpy array

    # Fill
    plt.imshow(cfmtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    # Ticks
    tick_marks_x = np.arange(len(xticks_ref))
    tick_marks_y = np.arange(len(yticks_ref))
    plt.xticks(tick_marks_x, xticks_ref, rotation=90)
    plt.yticks(tick_marks_y, yticks_ref)
    # Text
    fmt = '.2f' if normalize else 'd'
    thres = cfmtx.max() / 2.
    for i, j in itertools.product(range(cfmtx.shape[0]), range(cfmtx.shape[1])):
        #print(i, j)
        plt.text(j, i, int(cfmtx[i, j]), horizontalalignment='center', verticalalignment='center', color='white' if cfmtx[i, j] > thres else 'black')
    plt.tight_layout()
    # Label
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":

    metadata_path = 'dataset/metadata_box_case4t-x.csv'  # where is meta-data?
    label_column_name = 'category'  # @metdata.csv, which index indicate category?

    # Get label dictionary from the metadata
    # - Dictionary
    traindata_dict = dict(metadata_path=metadata_path, label_column_name=label_column_name, split='training')
    testdata_dict = dict(metadata_path=metadata_path, label_column_name=label_column_name, split='test')
    # - Dictionary/label
    traindata_dict_label = traindata_dict.label_dict
    testdata_dict_label = testdata_dict.label_dict


#1. A simple test
    #               (predicted classes)
    #  (Actual   |   0     1      2
    #  classes)  |----------------
    #      0     |   0     0      0
    #      1     |   0     2      0
    #      2     |   0     1      0

    #cf = cfmtx([1, 1, 2], [1, 1, 1], 3, 3)
    #print(cf), plt.imshow(cf), plt.show()

#2. Old; it need to be updated
    # Plot a confusion matrix from multiple files in .csv
    #c1 = merge(folder='result/tran_nfrz_mspecdb15', dim=39 , keyword='cfm', n_data_per_category=50, normalize=False)
    #c2 = merge(folder='result/tr_nf_mspdb_2048_2048_592_ep50', dim=39, keyword='cfm', n_data_per_category=50, normalize=False)
    #draw(cfmtx=c2, normalize=False)

#3.
    # Draw a confusion matrix whose true label = prediction label; e.g. Draw a cfmtx from a validation
    draw(file='result/case4t-2/case4t-2-merged.csv', normalize=True, ticks_ref=list(traindata_dict_label.values()))

#4.
    # Draw a confusion matrix whose true label ~= prediction label; e.g. Draw a cfmtx from a test
    #draw2(file='result/case0-0/test_case0-0.csv', normalize=True, xticks_ref=list(traindata_dict_label.values()), yticks_ref=list(testdata_dict_label.values()))
