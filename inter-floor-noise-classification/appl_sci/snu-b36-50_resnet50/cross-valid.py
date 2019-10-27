"""
Descriptions
    "cross-valid.py"
"""

# Public python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import random as rnd

# Tensorflow slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim

# Custom Pythone modules
from gen_data import generate
from load_data import load
from cfmtx import cfmtx

pp  = {
    'gpu_device':'device:GPU:0',
    'dtype':tf.float32,

    'gen_data':False,                                            # Data need to be generated at least one time
    'saver':False,
    'freeze_layer':False,
    'pretrain_weights':'resnet_v1_50.ckpt',                      # Name of the pre-trained weights
    'bn': False,

    'random_search_cnt_max':1,
    'learning_rate_range':[np.log10(0.0001),np.log10(100)],
    'penalty_range':[np.log10(0.0001),np.log10(100)],
    'generated_hyperparamter':'random_hyperparameter.csv',

    'rec_name_summary':'result/resnet_v1_50-type-cv-summary.csv', # 'type' <-> 'position'
    'rec_name':'result/resnet_v1_50-type',                        # 'type' <-> 'position'
    'rec_name_cfm':'result/resnet_v1_50-type',                    # 'type' <-> 'position'
    'saver_name':'saver_resnet_v1_50-position.ckpt',              #  Not used
    'metadata_path':'metadata_type.csv',                          # 'type' <-> 'position'
    'traindata_path':'train_type_3s_224_224',                     # 'type' <-> 'position'
    'validdata_path':'valid_type_3s_224_224',                     # 'type' <-> 'position'

    'label_column_name':'category',
    'num_fold':5,
    'n_category':5,                                               # '5' (type) <-> '9' (position)
    'batch_size_tr':39,
    'batch_size_val':10,
    'n_epoch':1                                                   # For random search, use '30'
}

# Generate random hyper-parameters with size of "random_search_cnt_max" (Disabled)
reg = []
lr = []
for cnt in range(pp['random_search_cnt_max']):
    lr.append(10**rnd.uniform(pp['learning_rate_range'][0],pp['learning_rate_range'][1]))
    reg.append(10**rnd.uniform(pp['penalty_range'][0],pp['penalty_range'][1]))
"""
# Import the generated random hyperparameters
df = pd.read_csv(pp['generated_hyperparamter'])
lr = df['lr']
reg = df['reg']
"""
# Cross-validaton summary: [lr, reg, valid_acc_k1, valid_acc_k2, valid_acc_k3, valid_acc_k4, valid_acc_k5]
cv_summary = np.zeros([pp['random_search_cnt_max'],pp['num_fold']+2])

# Five fold
for fold in range(1, pp['num_fold']+1):
    # If "gen_data" = True, convert audio clips into pickle data format
    traindata_path = pp['traindata_path'] + '_k' + str(fold) + '.p'
    validdata_path = pp['validdata_path'] + '_k' + str(fold) + '.p'
    if pp['gen_data']:
        # Training data
        generate(metadata_path=pp['metadata_path'], data_path=traindata_path, is_traindata = True,
                 batch_size=pp['batch_size_tr'], label_column_name=pp['label_column_name'], split=str(fold))
        # Validation data
        generate(metadata_path=pp['metadata_path'], data_path=validdata_path, is_traindata = False,
                 batch_size=pp['batch_size_tr'], label_column_name=pp['label_column_name'], split = str(fold))
    else:
        pass

    # Mean(each channel of training data)
    patch_mean = np.array([0, 0, 0], np.float32)  # Init.
    dataframe = load(traindata_path, pp['batch_size_tr'])
    for i, row in dataframe.dataframe.iterrows():
        # Calculate mean of each channel
        patch = row['patch']
        patch_mean[0] += np.mean(patch[:, :, 0])  # Ch 0
        patch_mean[1] += np.mean(patch[:, :, 1])  # Ch 1
        patch_mean[2] += np.mean(patch[:, :, 2])  # Ch 2
        # print(patch_mean)
    patch_mean = patch_mean / len(dataframe.dataframe['patch'])
    print("patch_mean:", patch_mean)
    # Delete "dataframe" from the memory
    dataframe.left = None
    del dataframe

    # Load the training data (.p); Note that "dataframe" is an instance
    dataframe_tr = load(traindata_path, pp['batch_size_tr'])
    num_batch_tr = dataframe_tr.n_batch
    # Load the validation data (.p)
    dataframe_valid = load(validdata_path, pp['batch_size_val'])
    num_batch_valid = dataframe_valid.n_batch

    # Loop for hyper-parameter optimization via random search
    for cnt in range(pp['random_search_cnt_max']):
        learning_rate = lr[cnt]; penalty = reg[cnt]

        print('cnt:', cnt, 'fold:', fold, "lr:", learning_rate, "penalty:", penalty)

        # Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Placeholders
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3]) - tf.constant([[patch_mean[0], patch_mean[1], patch_mean[2]]], dtype=pp['dtype'], shape=[1, 1, 1, 3])
        y = tf.placeholder(tf.float32, [None, pp['n_category']])

        # VGG16 instance; Transfer the pretrained weights
        is_training=True
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(imgs, 1000, is_training=is_training)

            # Saver & restoration of the pre-trained weights
            saver = tf.train.Saver()
            saver.restore(sess, 'resnet_v1_50.ckpt')

            # Adaptation layer (0ld)
            net = tf.nn.relu(net)
            net = tf.reshape(net, [-1, 1000])
            fc_w = tf.Variable(tf.truncated_normal([1000, pp['n_category']], dtype=tf.float32, stddev=1e-2), trainable=True)
            fc_b = tf.Variable(tf.constant(1.0, shape=[pp['n_category']], dtype=tf.float32), trainable=True)
            net = tf.nn.bias_add(tf.matmul(net, fc_w), fc_b)
            init_new_vars_op = tf.variables_initializer([fc_w, fc_b])  # Var. to be initialized
            sess.run(init_new_vars_op)  # Init.

        # Logits, y_out, loss
        y_out = tf.nn.softmax(net)  # softmax(logits)
        l2_loss = penalty * tf.nn.l2_loss(fc_w)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y)) + l2_loss

        # Accuracy measurement
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, pp['dtype']))

        # Optimizer
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        #Training and validation #

        # Variables used for recording training and validation
        rec_epoch = []
        rec_train_err = []
        rec_train_acc = []
        rec_valid_err = []
        rec_valid_acc = []

        # When the current validation accuracy is the best, update the followings
        best_valid_flag = True
        best_valid_epoch = 0
        best_valid_acc = 0
        best_cfm = []

        print("Start training...")

        # Training Loop; iter = epoch
        for epoch in range(pp['n_epoch']):
            # Variables for calculating average error and average accuracy
            aver_train_err = 0
            aver_train_acc = 0
            pp['is_training']=True
            # Mini-batch training
            for i in range(num_batch_tr):
                batch_x, batch_y = dataframe_tr.next_batch()
                err, acc, _ = sess.run([loss, accuracy, train_op], feed_dict={imgs: batch_x, y: batch_y})
                aver_train_err += err
                aver_train_acc += acc
            aver_train_err = aver_train_err / num_batch_tr
            aver_train_acc = aver_train_acc / num_batch_tr
            print("epoch:", epoch, "av_tr_err:", aver_train_err, "av_tr_acc:", aver_train_acc)

            # Validation
            # Variables for calculating average-error and average-accuracy
            aver_valid_err = 0
            aver_valid_acc = 0
            cfm = np.zeros([pp['n_category'], pp['n_category']])  # Initialize a confusion matrix

            # Mini-batch validation
            net2 = net
            y_out2 = tf.nn.softmax(net2)  # softmax(logits)
            for i in range(num_batch_valid):
                # Accuracy measurement
                correct_prediction = tf.equal(tf.argmax(y_out2, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, pp['dtype']))

                batch_x, batch_y = dataframe_valid.next_batch()
                acc = sess.run(accuracy, feed_dict={imgs: batch_x, y: batch_y})
                # - For confusion mtx
                prob = sess.run(y_out2, feed_dict={imgs: batch_x})  # Probability
                preds = (np.argmax(prob, axis=1))  # Predictions
                label = (np.argmax(batch_y, axis=1))  # Labels
                # print(preds)
                # print(label)
                cfm = cfm + cfmtx(label, preds, pp['n_category'], pp['batch_size_val'])  # Update confusion matrix
                aver_valid_err += err
                aver_valid_acc += acc
            aver_valid_err = aver_valid_err / num_batch_valid
            aver_valid_acc = aver_valid_acc / num_batch_valid
            print("av_val_err:", aver_valid_err, "av_val_acc:", aver_valid_acc)
            # If "aver_valid_acc" > "best_valid_acc"
            if (aver_valid_acc > best_valid_acc):
                best_valid_flag = True
                best_valid_epoch = epoch
                best_valid_acc = aver_valid_acc
                best_cfm = cfm
                print("The confusion matrix is saved...")
            else:
                best_valid_flag = False
            # if Nan or Inf, skip the model
            if np.isnan(aver_train_err) | np.isinf(aver_train_err): break; print("skip this model")
            # Record accuracy
            rec_epoch.append(epoch)
            rec_train_err.append(aver_train_err)
            rec_train_acc.append(aver_train_acc)
            rec_valid_err.append(aver_valid_err)
            rec_valid_acc.append(aver_valid_acc)
            # Delete "net2"
            net2 = None; del net2


        # Save values in the recording variables
        record = pd.DataFrame()
        record['epoch'] = rec_epoch
        record['train_err'] = rec_train_err
        record['train_acc'] = rec_train_acc
        record['valid_err'] = rec_valid_err
        record['valid_acc'] = rec_valid_acc
        record.to_csv(pp['rec_name'] + '-h' + str(cnt) + '-f' + str(fold) + '.csv')

        # Save the confusion matrix
        record_cfm = pd.DataFrame(cfm)
        record_cfm.to_csv(pp['rec_name_cfm'] + '-h' + str(cnt) + '-f' + str(fold) + '-ep' + str(best_valid_epoch) + '.csv')

        # Record cross-validation summary: [lr, reg, valid_acc_k1, valid_acc_k2, valid_acc_k3, valid_acc_k4, valid_acc_k5]
        cv_summary[cnt][0] = learning_rate
        cv_summary[cnt][1] = penalty
        cv_summary[cnt][fold+1] = best_valid_acc
        #print(cv_summary)

        # Reset graph for validation of the next model
        tf.reset_default_graph()
        dataframe_tr.left = None
        dataframe_valid.left = None
        net = None; del net
        end_points = None; del end_points

record_summary = pd.DataFrame(cv_summary)
record_summary.to_csv(pp['rec_name_summary'])