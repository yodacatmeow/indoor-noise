"""
Descriptions
    "cross-valid.py"
    Prior to run this code, run "dataset.py" to generate dataset

Issues
    dtype
"""

# Python modules
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import random as rnd
import os

# Custom Python modules
import vgg16_adap
from load_data import load
from cfmtx import cfmtx

# default parameters
pp  = {
    'gpu_device': '0',
    'dtype': tf.float32,

    'saver': True,
    'freeze_layer': False,
    'pretrain_weights':'../0_model_weights/vgg16_imagenet_1000.npz',               # Trained weights on ImageNet
    'bn': False,                                                # Batch normalization

    'is_optimal_h': True,                                      # "False" for hyper-parameter searching
    'cnt_max_rand_search': 1,                                  # Set as 100 for random hyper-parameter search
    'learning_rate_interval':[np.log10(0.0001),np.log10(100)],  # Learning-rate search range
    'penalty_interval':[np.log10(0.0001),np.log10(100)],        # Strength of regularization search range
    'optimal_lr': 0.00259636269237533,
    'optimal_reg':0.00541630894479606,

    'name_rec_summary':'result/summary.csv',                    # Training summary
    'name_rec':'result/loss',                                   # Record loss
    'name_rec_cfm':'result/cfm',                                # Record confusion matrix
    'name_saver':'saver',

    'path_metadata':'metadata.csv',
    'path_data_train':'train',
    'path_data_valid':'valid',

    'labels':'position',
    'n_fold': 5,
    'size_mbatch_train': 64,
    'size_mbatch_valid': 1,
    'n_epoch': 50                                                # For random search, set this as '30'
}

# Optimal hyper-parameters
optimal_lr = pp['optimal_lr']
optimal_reg = pp['optimal_reg']

# Select a GPU device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"]=pp['gpu_device']

# Generate random hyper-parameters
reg = []; lr = []                                          # reg = regularization strength; lr = learning rate
for cnt in range(pp['cnt_max_rand_search']):
    lr.append(10**rnd.uniform(pp['learning_rate_interval'][0],pp['learning_rate_interval'][1]))
    reg.append(10**rnd.uniform(pp['penalty_interval'][0],pp['penalty_interval'][1]))

# Cross-validaton summary: [lr, reg, valid_acc_k1, valid_acc_k2, valid_acc_k3, valid_acc_k4, valid_acc_k5]
cv_summary = np.zeros([pp['cnt_max_rand_search'],pp['n_fold']+2])

# 5-fold cross-validation:
for fold in range(1, pp['n_fold'] + 1):
    print("fold:", fold)
    # training data; validation data
    data_train = load(pp['path_data_train'] + '_k' + str(fold) + '.p', pp['size_mbatch_train'], pp['labels'])
    data_valid = load(pp['path_data_valid'] + '_k' + str(fold) + '.p', pp['size_mbatch_valid'], pp['labels'])

    # Get the global mean of the training data
    patch_mean = np.array([0, 0, 0], np.float32)

    # Disabled
    #patch_mean[0] = patch_mean[1] = patch_mean[2] = np.float32(data_train.global_mean())

    # Hyper-parameter optimization using random search
    for cnt in range(pp['cnt_max_rand_search']):
        # A random learning-rate and a random regularization strength
        lr_rand = lr[cnt]; reg_rand = reg[cnt]
        if pp['is_optimal_h']: lr_rand = optimal_lr; reg_rand = optimal_reg;
        print('cnt:', cnt, 'fold:', fold, "random learning-rate:", lr_rand, ", random regularization strength:", reg_rand)

        # TensorFlow Session
        config = tf.ConfigProto(); config.gpu_options.allow_growth = True; sess = tf.Session(config=config)

        # Placeholders
        imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # [None, width_VGG16 * height_VGG16 * depth_VGG16]
        y = tf.placeholder(tf.float32, [None, data_train.n_category])
        # Update the global mean
        vgg = vgg16_adap.vgg16(imgs=imgs, img_mean=np.array([0, 0, 0]), weights=pp['pretrain_weights'], sess=sess, bn=pp['bn'], bn_is_training=False, num_output=data_train.n_category)
        vgg.mean = patch_mean

        # Logits, y_out, loss <- hyperparameter dependent
        logits = vgg.fc4l; y_out = tf.nn.softmax(logits)
        l2_loss_4w = reg_rand * tf.nn.l2_loss(vgg.fc4w)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + l2_loss_4w

        # Accuracy measurement
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Optimizer
        if pp['freeze_layer']: train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_rand).minimize(loss, var_list=[vgg.fc1w,vgg.fc1b,vgg.fc2w,vgg.fc2b,vgg.fc3w,vgg.fc3b,vgg.fc4w,vgg.fc4b])
        else:train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_rand).minimize(loss)

        # New FCs need to be initialized
        init_new_vars_op = tf.variables_initializer([vgg.fc4w, vgg.fc4b]); sess.run(init_new_vars_op)

        # Variables for recording training and validation
        rec_epoch = []; rec_train_err = []; rec_train_acc = []; rec_valid_err = []; rec_valid_acc = []

        # When the current validation accuracy is the highest value, update the followings
        best_valid_flag = True; best_valid_epoch = 0; best_valid_acc = 0; best_cfm = []

        # Training via mini-batch gradient descent; validation
        for epoch in range(pp['n_epoch']):
            # Average error and average accuracy for training
            aver_train_err = 0; aver_train_acc = 0; vgg.bn_is_training = True

            # Training loop
            for i in range(data_train.n_batch):
                batch_x, batch_y = data_train.next_batch()
                err, acc, _ = sess.run([loss, accuracy, train_op], feed_dict={vgg.imgs: batch_x, y: batch_y})
                aver_train_err += err; aver_train_acc += acc
            aver_train_err = aver_train_err / data_train.n_batch; aver_train_acc = aver_train_acc / data_train.n_batch

            print("epoch:", epoch, "av_tr_err:", aver_train_err, "av_tr_acc:", aver_train_acc)

            # Average-error and average-accuracy for validation
            aver_valid_err = 0; aver_valid_acc = 0; vgg.bn_is_training = False
            # Initialize a confusion matrix
            cfm = np.zeros([data_train.n_category, data_train.n_category])

            # Validation loop
            for i in range(data_valid.n_batch):
                batch_x, batch_y = data_valid.next_batch()
                err, acc = sess.run([loss, accuracy], feed_dict={vgg.imgs: batch_x, y: batch_y})
                # Predicted pseudo probability
                prob = sess.run(vgg.probs, feed_dict={vgg.imgs: batch_x})
                # Prediction; True label
                preds = (np.argmax(prob, axis=1)); label = (np.argmax(batch_y, axis=1))
                # Updata the confusion matrix
                cfm = cfm + cfmtx(label=label, prediction=preds, dim=data_train.n_category, batch_size=data_valid.batch_size)
                aver_valid_err += err; aver_valid_acc += acc;
            aver_valid_err = aver_valid_err / data_valid.n_batch; aver_valid_acc = aver_valid_acc / data_valid.n_batch

            print("epoch:", epoch, "av_val_err:", aver_valid_err, "av_val_acc:", aver_valid_acc)

            # If "aver_valid_acc" > "best_valid_acc"
            if (aver_valid_acc > best_valid_acc): best_valid_flag = True; best_valid_epoch = epoch; best_valid_acc = aver_valid_acc; best_cfm = cfm
            else:best_valid_flag = False

            # Save weights
            if pp['saver'] and best_valid_flag: vgg.save_weights(pp['name_saver'] + '_k' + str(fold) + '.npz', sess)
            else: pass

            # Record via appending
            rec_epoch.append(epoch); rec_train_err.append(aver_train_err); rec_train_acc.append(aver_train_acc); rec_valid_err.append(aver_valid_err); rec_valid_acc.append(aver_valid_acc);

            # if Nan or Inf, skip the model
            if np.isnan(aver_train_err) | np.isinf(aver_train_err): print("skip this model"); break;

        # Save values in the recording variables
        record = pd.DataFrame()
        record['epoch'] = rec_epoch; record['train_err'] = rec_train_err; record['train_acc'] = rec_train_acc; record['valid_err'] = rec_valid_err; record['valid_acc'] = rec_valid_acc
        record.to_csv(pp['name_rec'] + '-h' + str(cnt) + '-f' + str(fold) + '.csv')

        # Save the confusion matrix
        record_cfm = pd.DataFrame(best_cfm)
        record_cfm.to_csv(pp['name_rec_cfm'] + '-h' + str(cnt) + '-f' + str(fold) + '-ep' + str(best_valid_epoch) + '.csv')

        # Record cross-validation summary: [lr, reg, valid_acc_k1, valid_acc_k2, valid_acc_k3, valid_acc_k4, valid_acc_k5]
        cv_summary[cnt][0] = lr_rand; cv_summary[cnt][1] = reg_rand; cv_summary[cnt][fold + 1] = best_valid_acc

        # Clears the default graph stack and resets the global default graph
        tf.reset_default_graph()
        sess = vgg = None
        del vgg, sess

record_summary = pd.DataFrame(cv_summary)
record_summary.to_csv(pp['name_rec_summary'])