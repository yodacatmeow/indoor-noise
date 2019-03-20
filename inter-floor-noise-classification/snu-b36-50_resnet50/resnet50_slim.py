"""
ResNet50 using TensorFlow slim

References
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py
    https://stackoverflow.com/questions/48947083/re-train-pre-trained-resnet-50-model-with-tf-slim-for-classification-purposes

    pre-trained weights
        https://github.com/tensorflow/models/tree/master/research/slim#Tuning

"""

from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imread, imresize
from caffe_classes import class_names

batch_size = 1
height, width, channels = 224, 224, 3

# Test input image
img1 = imread('laska.png', mode='RGB')
img1 = imresize(img1, (224, 224))
img1 = img1.reshape(1,224,224,3)

# Create graph
inputs = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits, end_points = resnet_v1.resnet_v1_50(inputs, 1000, is_training=False)

    saver = tf.train.Saver()

    # Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Restore the pre-trained weights
    saver.restore(sess, 'resnet_v1_50.ckpt')

    #predict_values, logit_values = sess.run([end_points, logits], feed_dict= {inputs: img1})

    # Probability
    probs = tf.nn.softmax(logits)  # New
    prob = sess.run(probs, feed_dict={inputs: img1})[0][0][0] # Probability
    #print(prob)
    preds = (np.argsort(prob)[::-1])[0:5]
    #print(preds)
    for p in preds:
        print(class_names[p], prob[p])
