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

# Custom Pythone modules
import alexnet_adap
from gen_data import generate
from load_data import load
from cfmtx import cfmtx

# default parameters
# - Best learning-rate (type):
# - Best learning-rate (position): 0.0188400538952988
# - Best L2-penalty (type):
# - Best L2-Penalty (position): 0.000188739557416965
# default parameters
pp  = {
    'gpu_device':'device:GPU:0',
    'dtype':tf.float32,

    'gen_data':False,
    'saver':False,
    'is_transfer_learn':True,
    'freeze_layer':False,
    'pretrain_weights':'bvlc_alexnet.npy',  # Not used; import the weight @"alexnet_adap.py"

    'random_search_cnt_max':1,  # For random search, use 100
    'learning_rate_range':[np.log10(0.0188400538952988),np.log10(0.0188400538952988)],     # For hyperparameter search, np.log10(0.0001),np.log10(100)
    'penalty_range':[np.log10(0.000188739557416965),np.log10(0.000188739557416965)],       # For hyperparameter search, np.log10(0.0001),np.log10(100)

    'rec_name_summary':'result/alexnet-pos-0-cv-summary.csv',
    'rec_name':'result/alexnet-pos-0',
    'rec_name_cfm':'result/alexnet-pos-0',
    'saver_name':'saver_alexnet-pos-0',
    'metadata_path':'../dataset/metadata_type.csv',
    'traindata_path':'../dataset/train_pos_3s_227_227',
    'validdata_path':'../dataset/valid_pos_3s_227_227',

    'label_column_name':'category',
    'num_fold':5,
    'n_category':9,
    'batch_size_tr':39,
    'batch_size_val':10,
    'n_epoch':50    # For random search, use 30
}

# Generate random hyper-parameters with size of "random_search_cnt_max"
reg = []
lr = []
for cnt in range(pp['random_search_cnt_max']):
    lr.append(10**rnd.uniform(pp['learning_rate_range'][0],pp['learning_rate_range'][1]))
    reg.append(10**rnd.uniform(pp['penalty_range'][0],pp['penalty_range'][1]))

with tf.device(pp['gpu_device']):
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
            print('cnt:', cnt, 'fold:', fold, "learning_rate:", learning_rate, "penalty:", penalty)

            # Session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            # Placeholders
            imgs = tf.placeholder(tf.float32, [None, 227, 227, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]
            y = tf.placeholder(tf.float32, [None, pp['n_category']])

            # AlexNet instance; Transfer the pretrained weights
            alexnet = alexnet_adap.alexnet(imgs=imgs, img_mean=patch_mean, weights=pp['pretrain_weights'], sess=sess, num_output=pp['n_category'])
            # One should initialize "FCb" graph for TensorFlow; "alexnet_adap.py" includes global_variables_initializer()
            sess.run(tf.global_variables_initializer())

            # Logits, y_out, loss
            logits = alexnet.fc9
            y_out = tf.nn.softmax(logits)
            l2_loss_9w = penalty * tf.nn.l2_loss(alexnet.fc9w)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + l2_loss_9w

            # Accuracy measurement
            correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Optimization
            if pp['freeze_layer']:
                train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, var_list=[alexnet.fc6w, alexnet.fc6b, alexnet.fc7w, alexnet.fc7b,
                                                                                                          alexnet.fc8w, alexnet.fc8b, alexnet.fc9w, alexnet.fc9b])
            # Update all layers
            else:
                train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

            # One should initialize "FCb" graph for TensorFlow
            # In the case of not transferring the pre-trained weights, we need to initialize the whole graph
            if pp['is_transfer_learn']:
                init_new_vars_op = tf.variables_initializer([alexnet.fc9w, alexnet.fc9b])  # New; New FC layer @"vgg16_adap" needs graph initialization
                sess.run(init_new_vars_op)  # New; Run graph initialization
            else:
                sess.run(tf.global_variables_initializer())

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

            # Loop; iter = epoch
            for epoch in range(pp['n_epoch']):
                # Variables for calculating average error and average accuracy
                aver_train_err = 0
                aver_train_acc = 0

                # Mini-batch training
                for i in range(num_batch_tr):
                    batch_x, batch_y = dataframe_tr.next_batch()
                    err, acc, _ = sess.run([loss, accuracy, train_op], feed_dict={alexnet.imgs: batch_x, y: batch_y})
                    aver_train_err += err
                    aver_train_acc += acc
                aver_train_err = aver_train_err / num_batch_tr
                aver_train_acc = aver_train_acc / num_batch_tr
                print("epoch:", epoch, "av_tr_err:", aver_train_err, "av_tr_acc:", aver_train_acc)

                # Variables for calculating average-error and average-accuracy
                aver_valid_err = 0
                aver_valid_acc = 0
                cfm = np.zeros([pp['n_category'], pp['n_category']])  # Initialize a confusion matrix

                # Mini-batch validation
                for i in range(num_batch_valid):
                    batch_x, batch_y = dataframe_valid.next_batch()
                    err, acc = sess.run([loss, accuracy], feed_dict={alexnet.imgs: batch_x, y: batch_y})

                    # - For confusion mtx
                    prob = sess.run(alexnet.probs, feed_dict={alexnet.imgs: batch_x})   # Probability
                    preds = (np.argmax(prob, axis=1))                           # Predictions
                    label = (np.argmax(batch_y, axis=1))                        # Labels
                    # print(preds)
                    # print(label)
                    cfm = cfm + cfmtx(label, preds, pp['n_category'], pp['batch_size_val'])  # Update confusion matrix

                    aver_valid_err += err
                    aver_valid_acc += acc
                aver_valid_err = aver_valid_err / num_batch_valid
                aver_valid_acc = aver_valid_acc / num_batch_valid
                print("epoch:", epoch, "av_val_err:", aver_valid_err, "av_val_acc:", aver_valid_acc)
                # If "aver_valid_acc" > "best_valid_acc"
                if (aver_valid_acc > best_valid_acc):
                    best_valid_flag = True
                    best_valid_epoch = epoch
                    best_valid_acc = aver_valid_acc
                    best_cfm = cfm
                    print("The confusion matrix is saved...")
                else:
                    best_valid_flag = False

                # Save weights
                if pp['saver'] and best_valid_flag:
                    alexnet.save_weights(pp['saver_name']+'_k' + str(fold)+'.npz', sess)
                    print("The weights are saved...")
                else:
                    pass

                # Record via appending
                rec_epoch.append(epoch)
                rec_train_err.append(aver_train_err)
                rec_train_acc.append(aver_train_acc)
                rec_valid_err.append(aver_valid_err)
                rec_valid_acc.append(aver_valid_acc)

                # if Nan or Inf, skip the model
                if np.isnan(aver_train_err) | np.isinf(aver_train_err): break; print("skip this model")

            # Save values in the recording variables
            record = pd.DataFrame()
            record['epoch'] = rec_epoch
            record['train_err'] = rec_train_err
            record['train_acc'] = rec_train_acc
            record['valid_err'] = rec_valid_err
            record['valid_acc'] = rec_valid_acc
            record.to_csv(pp['rec_name'] + '-h' + str(cnt) + '-f' + str(fold) + '.csv')

            # Save the confusion matrix
            record_cfm = pd.DataFrame(best_cfm)
            record_cfm.to_csv(pp['rec_name_cfm'] + '-h' + str(cnt) + '-f' + str(fold) + '-ep' + str(best_valid_epoch) + '.csv')

            # Record cross-validation summary: [lr, reg, valid_acc_k1, valid_acc_k2, valid_acc_k3, valid_acc_k4, valid_acc_k5]
            cv_summary[cnt][0] = learning_rate
            cv_summary[cnt][1] = penalty
            cv_summary[cnt][fold+1] = best_valid_acc
            #print(cv_summary)

            # Reset graph for validation of the next model
            tf.reset_default_graph()
            alexnet = None; del alexnet
            sess = None; del sess

    record_summary = pd.DataFrame(cv_summary)
    record_summary.to_csv(pp['rec_name_summary'])