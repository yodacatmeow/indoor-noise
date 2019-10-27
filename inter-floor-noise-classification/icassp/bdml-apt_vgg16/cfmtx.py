"""
"cfmtx.py"

confusion matrix function
"""
# Public python modules#
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

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
