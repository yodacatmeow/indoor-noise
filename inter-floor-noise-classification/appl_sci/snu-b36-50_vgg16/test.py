# Public python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Custom Pythone modules
import vgg16_adap
import cfmtx
from gen_data import generate
from gen_data import dict
from load_data import load
from cfmtx import cfmtx2

pp = {
    'gpu_device':'device:GPU:0',
    'gen_data':False,
    'bn': False,

    'rec_name':'result/vgg16-type-0-test',
    'pretrain_weights':'saver_vgg16-type-0',
    'metadata_path':'../dataset/metadata_type.csv',
    'traindata_path':'../dataset/train_type_3s_224_224',
    'testdata_path':'../dataset/test_type_3s_224_224.p',

    'label_column_name': 'category',
    'num_fold': 5,
    'n_category': 5,
    'batch_size_tr': 10,
    'batch_size_test': 1
}

with tf.device(pp['gpu_device']):
    # Generate a test dataset
    if pp['gen_data']:
        testdata_gen = generate(metadata_path = pp['metadata_path'], data_path = pp['testdata_path'], is_traindata = False,
                                batch_size = pp['batch_size_test'], label_column_name = pp['label_column_name'], split = 'test')
    else:
        pass

    # Get a label dictionary from the metadata
    traindata_dict = dict(metadata_path = pp['metadata_path'], label_column_name = pp['label_column_name'], split = '1')    # Use Fold 1
    testdata_dict = dict(metadata_path = pp['metadata_path'], label_column_name = pp['label_column_name'], split = 'test')
    # Dictionary
    traindata_dict_label = traindata_dict.label_dict
    testdata_dict_label = testdata_dict.label_dict
    print("Training data dictionary:", traindata_dict_label, "length:", len(traindata_dict_label))
    print("Test data dictionary:", testdata_dict_label, "length:", len(testdata_dict_label))

    for fold in range(5, pp['num_fold']+1):
        traindata_path = pp['traindata_path'] + '_k' + str(fold) + '.p'
        # Calculate mean of each channel of training data
        patch_mean = np.array([0, 0, 0], np.float32)  # Init.
        dataframe = load(traindata_path, pp['batch_size_tr'])  # Instance
        for i, row in dataframe.dataframe.iterrows():
            #- Calculate mean of each channel
            patch = row['patch']
            patch_mean[0] += np.mean(patch[:, :, 0])  # Ch 0
            patch_mean[1] += np.mean(patch[:, :, 1])  # Ch 1
            patch_mean[2] += np.mean(patch[:, :, 2])  # Ch 2
            # print(patch_mean)
        patch_mean = patch_mean / len(dataframe.dataframe['patch'])
        print("patch_mean:", patch_mean)
        #- Delete "dataframe" from the memory
        dataframe.left = None

        # Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Placeholders
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]

        # VGG16 instance; Transfer the pretraining weights
        vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=patch_mean, weights=pp['pretrain_weights']+'_k' + str(fold) + '.npz',
                               sess=sess, bn=pp['bn'], bn_is_training=False, num_output=pp['n_category'])

        # Logits, y_out, loss
        logits = vgg.fc4l
        y_out = tf.nn.softmax(logits)

        # Loading validation data (.p)
        dataframe_test = load(pp['testdata_path'], pp['batch_size_test'])
        num_batch_test = dataframe_test.n_batch

        print("Start test...")

        cfm = np.zeros([len(testdata_dict_label), len(traindata_dict_label)])
        # Test
        for i in range(num_batch_test):
            batch_x, batch_y = dataframe_test.next_batch()
            #- For confusion mtx
            prob = sess.run(vgg.probs, feed_dict={vgg.imgs: batch_x})[0]  # Probability

            #- Top 1
            predict_digit = (np.argsort(prob)[::-1])[0]
            label_digit = np.argmax(batch_y)
            print("True label:",label_digit, "predicted label:",predict_digit)

            #- Update confusion matrix
            cfm = cfm + cfmtx2(label_digit, predict_digit, cfm.shape)


        # Save values in the recording variables
        record_cfm = pd.DataFrame(cfm)
        record_cfm.to_csv(pp['rec_name']+'_k' + str(fold) + '.csv')

        # Draw cfm
        print(list(traindata_dict_label.values()))
        #cfmtx.draw2(file='result/vgg-loc-2-test_cfm.csv', normalize=True, xticks_ref=list(traindata_dict_label.values()), yticks_ref=list(testdata_dict_label.values()))
