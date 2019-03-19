"""
Description
    # AlexNet implementation with TensorFlow
    # Michael's implementation is adapted under class "alexnet"

References
    # Michael Geurzhoy and Davi Frossard, 2016
        https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    # Frossard's "vgg16.py" style
        http://www.cs.toronto.edu/~frossard/post/vgg16/
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from caffe_classes import class_names

class alexnet:
    def __init__(self, imgs, img_mean, weights=None, sess=None, num_output=None):
        self.imgs = imgs
        self.img_mean = img_mean
        self.weights = weights
        self.num_output=num_output
        self.sess = sess
        self.neuralnets()
        self.probs = tf.nn.softmax(self.fc9)
        # When testing this module with an image, use the following logits
        #self.probs = tf.nn.softmax(self.fc8)


    def neuralnets(self):
        self.parameters = []
        # Load pre-training parameters
        net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
        #net_data = np.load(open(self.weights, "rb"), encoding="latin1").item()
        print("bvlc_alex.npy is loaded...")

        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow'''
            c_i = input.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
                kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([[self.img_mean[0], self.img_mean[0], self.img_mean[0]]], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        with tf.name_scope('conv1') as scope:
            group = 1; k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4;
            kernel = tf.Variable(net_data["conv1"][0], trainable=True, name='weights')
            biases = tf.Variable(net_data["conv1"][1], trainable=True, name='biases')
            conv1 = conv(images, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv1 = tf.nn.relu(conv1, name=scope)
            self.parameters += [kernel, biases]

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        with tf.name_scope('lrn1') as scope:
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0;
            self.lrn1 = tf.nn.local_response_normalization(self.conv1, depth_radius=radius, alpha=alpha, beta=beta,bias=bias)

        # Maxpool1
        k_h = 3; k_w = 3; s_h = 2; s_w = 2;
        self.pool1 = tf.nn.max_pool(self.lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID', name='pool1')

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        with tf.name_scope('conv2') as scope:
            group = 2; k_h = 5; k_w = 5; c_i = int(96 / group); c_o = 256; s_h = 1; s_w = 1;
            kernel = tf.Variable(net_data["conv2"][0], trainable=True, name='weights')
            biases = tf.Variable(net_data["conv2"][1], trainable=True, name='biases')
            conv2 = conv(self.pool1, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv2 = tf.nn.relu(conv2, name=scope)
            self.parameters += [kernel, biases]

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        with tf.name_scope('lrn2') as scope:
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0;
            self.lrn2 = tf.nn.local_response_normalization(self.conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

        # Maxpool2
        k_h = 3; k_w = 3; s_h = 2; s_w = 2;
        self.pool2 = tf.nn.max_pool(self.lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID', name='pool2')

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        with tf.name_scope('conv3') as scope:
            group = 1; k_h = 3; k_w = 3; c_i = int(256 / group); c_o = 384; s_h = 1; s_w = 1;
            kernel = tf.Variable(net_data["conv3"][0], trainable=True, name='weights')
            biases = tf.Variable(net_data["conv3"][1], trainable=True, name='biases')
            conv3 = conv(self.pool2, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv3 = tf.nn.relu(conv3, name=scope)
            self.parameters += [kernel, biases]

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        with tf.name_scope('conv4') as scope:
            group = 2; k_h = 3; k_w = 3; c_i = int(384 / group); c_o = 384; s_h = 1; s_w = 1;
            kernel = tf.Variable(net_data["conv4"][0], trainable=True, name='weights')
            biases = tf.Variable(net_data["conv4"][1], trainable=True, name='biases')
            conv4 = conv(self.conv3, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv4 = tf.nn.relu(conv4, name=scope)
            self.parameters += [kernel, biases]

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        with tf.name_scope('conv5') as scope:
            group = 2; k_h = 3; k_w = 3; c_i = int(384 / group); c_o = 256; s_h = 1; s_w = 1;
            kernel = tf.Variable(net_data["conv5"][0], trainable=True, name='weights')
            biases = tf.Variable(net_data["conv5"][1], trainable=True, name='biases')
            conv5 = conv(self.conv4, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            self.conv5 = tf.nn.relu(conv5, name=scope)
            self.parameters += [kernel, biases]

        # Maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2;
        self.pool5 = tf.nn.max_pool(self.conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID', name='pool5')

        # FC6
        # fc(4096, name='fc6')
        with tf.name_scope('fc6') as scope:
            fc6w = tf.Variable(net_data["fc6"][0], trainable=True, name='weights')
            fc6b = tf.Variable(net_data["fc6"][1], trainable=True, name='biases')
            self.fc6 = tf.nn.relu_layer(tf.reshape(self.pool5, [-1, int(np.prod(self.pool5.get_shape()[1:]))]), fc6w, fc6b) # Flat
            self.parameters += [fc6w, fc6b]

        # fc7
        # fc(4096, name='fc7')
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(net_data["fc7"][0],trainable=True, name='weights')
            fc7b = tf.Variable(net_data["fc7"][1], trainable=True, name='biases')
            self.fc7 = tf.nn.relu_layer(self.fc6, fc7w, fc7b)
            self.parameters += [fc7w, fc7b]

        # fc8
        # fc(1000, relu=False, name='fc8')
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(net_data["fc8"][0],trainable=True, name='weights')
            fc8b = tf.Variable(net_data["fc8"][1], trainable=True, name='biases')
            self.fc8 = tf.nn.relu_layer(self.fc7, fc8w, fc8b)
            self.parameters += [fc8w, fc8b]

        # fc9; fca; an adaptation layer
        with tf.name_scope('fc9') as scope:
            self.fc9w = tf.Variable(tf.truncated_normal([1000, self.num_output], dtype=tf.float32, stddev=1e-2), name='weights')
            self.fc9b = tf.Variable(tf.constant(1.0, shape=[self.num_output], dtype=tf.float32), trainable=True, name='biases')
            self.fc9 = tf.nn.bias_add(tf.matmul(self.fc8, self.fc9w), self.fc9b)
            self.parameters += [self.fc9w, self.fc9b]

    # Save weights as .npz; Using this function, you can save the weights after training
    def save_weights(self, weights_file, sess):
        weights = sess.run(self.parameters)
        keys = ['conv1_W', 'conv1_b', 'conv2_W', 'conv2_b', 'conv3_W', 'conv3_b'
                'conv4_W', 'conv4_b', 'conv5_W', 'conv5_b', 'fc6_W', 'fc6_b',
                'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b', 'fc9_W', 'fc9_b']
        np.savez(weights_file, **{name: value for name, value in zip(keys, weights)})

    def load_weights(self, weight_file, sess, load_adap_layer=False):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print(keys)

        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))                                    # i = layer, k = entries of layer_i
            sess.run(self.parameters[i].assign(weights[k]))


# For testing this module
if __name__ == '__main__':

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 227, 227, 3])
    img_mean = [0, 0, 0]

    alexnet = alexnet(imgs, img_mean, sess=sess, num_output=1000)
    sess.run(tf.global_variables_initializer())

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (227, 227))

    # Change RGB to BGR
    img1 = img1 - np.mean(img1)
    img1[:, :, 0], img1[:, :, 2] = img1[:, :, 2], img1[:, :, 0]

    # Soft-max classfier
    prob = sess.run(alexnet.probs, feed_dict={alexnet.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

    # Test "save_weights()":
    #alexnet.save_weights("weights_out.npz", sess)
    # Test "load_weights()":
    #alexnet.load_weights("weights_out.npz", sess)
