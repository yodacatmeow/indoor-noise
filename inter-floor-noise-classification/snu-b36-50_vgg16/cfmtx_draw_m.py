"""
"cfmtx_draw_m.py"

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

#file='result/vgg16-loc-opt-2/vgg-loc-2-test_cfm-merged.csv'
#file='result/vgg16-type-opt-2/vgg16-type-2-test_cfm-merged.csv'
#file='result/vgg16-side-end-pos-0/vgg16-loc-end-0-test_merged.csv'
#file='../resnet/result/resnet_v1_50_position-0_tuning/optimal-model-trial-0/resnet_v1_50-position-0-h23-merged-percent.csv' # <- Optimal model for pos classification
#file='../alexnet/result/alexnet-pos-tuning-0/optimal_model/alexnet-pos-0-h55-merged-percent.csv'                            # <- Optimal model for pos classification
file='result/vgg16-pos-0-tuning/optimal-model-trial-0-weigth-saved/vgg16-pos-0-h0_merged_percent.csv'                             # <- Optimal model for pos classification
#file='result/vgg16-pos-0-tuning/optimal-model-trial-0-wander-prob/vgg16-pos-0-test_merged_percent.csv'
normalize = False
xticks_ref = ['1F0m', '1F6m', '1F12m', '2F0m', '2F6m', '2F12m', '3F0m', '3F6m', '3F12m']
#xticks_ref = ['1F0m', '1F12m', '2F0m', '2F12m', '3F0m', '3F12m']
yticks_ref = ['3F1m', '3F2m', '3F3m', '3F4m', '3F5m', '3F7m', '3F8m', '3F9m', '3F10m', '3F11m']
yticks_ref = ['1F0m', '1F6m', '1F12m', '2F0m', '2F6m', '2F12m', '3F0m', '3F6m', '3F12m']
#xticks_ref = ['CD', 'HD', 'HH', 'MB', 'VC']
#yticks_ref = ['HH', 'MB']
tick_marks_x = np.arange(len(xticks_ref))
tick_marks_y = np.arange(len(yticks_ref))
# Read  csv file
df = pd.read_csv(file, sep=',')
# Drop the first col.
df = df.drop(df.columns[0], axis=1)
cfmtx = np.array(df) # numpy array
print(cfmtx)

# Fill
plt.imshow(cfmtx, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
#plt.colorbar()
# Ticks
plt.xticks(tick_marks_x, xticks_ref, rotation=90)
plt.yticks(tick_marks_y, yticks_ref)
# Text
#fmt = '.2f' if normalize else 'd'
thres = cfmtx.max() / 2.
for i, j in itertools.product(range(cfmtx.shape[0]), range(cfmtx.shape[1])):
    #print(i, j)
    plt.text(j, i, (cfmtx[i, j]), horizontalalignment='center', verticalalignment='center', color='white' if cfmtx[i, j] > thres else 'black')
plt.tight_layout()

# Label
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Save image as high quality image
plt.savefig('cfm.png', format = 'png', dpi = 300)

plt.show()