"""
Description
    "test.py"
    Prior to run this code, run "dataset.py" to generate dataset
"""
# Public python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Custom Pythone modules
import vgg16_adap
import cfmtx
from load_data import load
from cfmtx import cfmtx2

pp = {
    'gpu_device':'device:GPU:0',

    'pretrain_weights':'saver_vgg16-type',
    'bn': False,

    'name_rec_test':'result/vgg16-type-test',

    'path_metadata':'metadata.csv',
    'path_data_train':'train_type_3s_224_224',
    'path_data_test':'test_type_3s_224_224',

    'labels': 'type',
    'n_fold': 5,
    'size_mbatch_train': 10,
    'size_mbatch_test': 1
}

with tf.device(pp['gpu_device']):
    data_test = load(pp['path_data_test'] + '.p', pp['size_mbatch_test'], pp['labels'])

    print("test data dict:", data_test.labels_dict)

    for fold in range(1, pp['n_fold'] + 1):
        # Get the global mean of the training data
        data_train = load(pp['path_data_train'] + '_k' + str(fold) + '.p', pp['size_mbatch_train'], pp['labels'])
        patch_mean = np.array([0, 0, 0], np.float32); patch_mean[0] = patch_mean[1] = patch_mean[2] = np.float32(data_train.global_mean())
        print(patch_mean)
        print(data_train.labels_dict)

        # TensorFlow Session
        config = tf.ConfigProto(); config.gpu_options.allow_growth = True; sess = tf.Session(config=config)

        # Placeholders
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]

        # Import the pre-trained weights to a VGG instance and update the global mean
        vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=np.array([0, 0, 0]), weights=pp['pretrain_weights']+'_k' + str(fold) + '.npz', sess=sess, bn=pp['bn'], bn_is_training=False, num_output=data_train.n_category)
        vgg.mean = patch_mean

        # Logits, y_out, loss
        logits = vgg.fc4l; y_out = tf.nn.softmax(logits)

        print("Start test...")

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