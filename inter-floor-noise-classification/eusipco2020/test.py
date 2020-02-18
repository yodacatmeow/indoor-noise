"""
Description
    "test.py"
    Prior to run this code, run "cross-valid.py" to train vgg16
"""

# Python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os

# Custom Python modules
import vgg16_adap
import cfmtx
from load_data import load
from cfmtx import cfmtx2

pp = {
    'gpu_device':'0',

    'knowledge':'saver',                # Knowledge to be used for testing
    'bn': False,                        # Batch normalization

    'name_rec_test':'result/test',      # Save loss

    'path_metadata':'metadata.csv',     # Metadata
    'path_data_train':'train',          # Training data
    'path_data_test':'test',            # Test data

    'labels': 'type',
    'n_fold': 5,
    'size_mbatch_train': 10,
    'size_mbatch_test': 1,
    #'train_category_n': 8               # Output dimension of the network (decided by # categories in a training data)
}

# Select a GPU device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]=pp['gpu_device']

data_test = load(pp['path_data_test'] + '.p', pp['size_mbatch_test'], pp['labels'])

print("test data dict:", data_test.labels_dict)

for fold in range(1, pp['n_fold'] + 1):
    # Get the global mean of the training data
    data_train = load(pp['path_data_train'] + '_k' + str(fold) + '.p', pp['size_mbatch_train'], pp['labels'])
    patch_mean = np.array([0, 0, 0], np.float32)

    # Disabled
    #patch_mean[0] = patch_mean[1] = patch_mean[2] = np.float32(data_train.global_mean())
    
    print(patch_mean)
    print(data_train.labels_dict)

    # TensorFlow Session
    config = tf.ConfigProto(); config.gpu_options.allow_growth = True; sess = tf.Session(config=config)

    # Placeholders
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]

    # Import the pre-trained weights to a VGG instance and update the global mean
    vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=np.array([0, 0, 0]), weights=pp['knowledge']+'_k' + str(fold) + '.npz', sess=sess, bn=pp['bn'], bn_is_training=False, num_output=data_train.n_category)
    #vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=np.array([0, 0, 0]), weights=pp['knowledge'] + '_k' + str(fold) + '.npz',sess=sess, bn=pp['bn'], bn_is_training=False, num_output=pp['train_category_n'])

    vgg.mean = patch_mean

    # Logits, y_out, loss
    logits = vgg.fc4l; y_out = tf.nn.softmax(logits)

    print("Start a test...")

    # Initialize a confusion matrix
    cfm = np.zeros([data_test.n_category, data_train.n_category])

    # Test
    for i in range(data_test.n_batch):
        batch_x, batch_y = data_test.next_batch()
        # For confusion mtx
        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: batch_x})[0]  # Probability

        # Top 1 prediction
        predict_digit = (np.argsort(prob)[::-1])[0]
        label_digit = np.argmax(batch_y)
        #print("True label:", label_digit, "predicted label:", predict_digit)

        # Update confusion matrix
        cfm = cfm + cfmtx2(label_digit, predict_digit, cfm.shape)

    # Save the confusion matrix
    record_cfm = pd.DataFrame(cfm)
    record_cfm.to_csv(pp['name_rec_test'] + '_k' + str(fold) + '.csv')

    # Clears the default graph stack and resets the global default graph
    tf.reset_default_graph()
    sess = vgg = None
    del vgg, sess