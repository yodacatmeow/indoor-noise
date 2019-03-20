"""
Description
    # VGG16_adap for supervised representation learning; 13 categories

References
    # David Frossard's model
    # "vgg16.py"
        http://www.cs.toronto.edu/~frossard/post/vgg16/
    # save weight in a loop
        https://stackoverflow.com/questions/30850702/name-numpys-keywords-with-savez-while-using-arbitrary-number-of-arguments
    # Batch normalization
        https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/layers/batch_norm
        http://openresearch.ai/t/topic/80
"""


import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from category import class_names

class vgg16:
    # "__init__" below initializes the object       # New
    def __init__(self, imgs, img_mean, weights=None, sess=None, bn=False, bn_is_training=True, num_output=None):
        self.imgs = imgs
        self.img_mean = img_mean                    # New
        self.bn = bn                                # New; A flag for batch normalization
        self.bn_is_training = bn_is_training
        self.num_output=num_output

        self.convlayers()
        self.fc_layers()
        #self.probs = tf.nn.softmax(self.fc3l)      # Dis-abled
        self.probs = tf.nn.softmax(self.fc4l)       # New


        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
            print("Weights", weights, "are loaded ...") # New

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')          # Dis-abled
            mean = tf.constant([[self.img_mean[0], self.img_mean[0], self.img_mean[0]]], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #- Batch normalization
            if self.bn:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), name='weights')  # Modified: fc1w -> self.fc1w
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')    # Modified: fc1b -> self.fc1w
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1w), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [self.fc1w, self.fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')   # Modified: fc2w -> self.fc2w
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')    # Modified: fc2b -> self.fc2w
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2w), self.fc2b)
            #- Batch normalization
            if self.bn:
                fc2l = tf.contrib.layers.batch_norm(fc2l, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [self.fc2w, self.fc2b]

        # fc3; Adaptation layer a; FCa; Modified;
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1), name='weights')  # Modified: fc3w -> self.fc3w
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32), trainable=True, name='biases')   # Modified: fc3b -> self.fc3b
            #self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)                        # Dis-abled
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)                    # New
            #- Batch normalization
            if self.bn:
                fc3l = tf.contrib.layers.batch_norm(fc3l, center=True, scale=False, is_training=self.bn_is_training)
            else:
                pass
            self.fc3l = tf.nn.relu(fc3l)                                                        # New
            self.parameters += [self.fc3w, self.fc3b]
        # FIXME: Do we need to initialize weights in this layer?? or start training from the current weight??

        # fc4; New; Adaptation layer a; FCb; Graph initialization (TensorFlow) allowed (self.~)
        with tf.name_scope('fc4') as scope:
            self.fc4w = tf.Variable(tf.truncated_normal([1000,self.num_output], dtype=tf.float32, stddev=1e-2), name='weights')
            self.fc4b = tf.Variable(tf.constant(1.0, shape=[self.num_output], dtype=tf.float32), trainable=True, name='biases')
            self.fc4l = tf.nn.bias_add(tf.matmul(self.fc3l, self.fc4w), self.fc4b)
            self.parameters += [self.fc4w, self.fc4b]
            #self.parameters += [fc4w, fc4b]                                        # Dis-abled

    def load_weights(self, weight_file, sess, load_adap_layer=False):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print(keys)

        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))                                    # i = layer, k = entries of layer_i
            sess.run(self.parameters[i].assign(weights[k]))


    # Save weights as .npz; Using this function, you can save the weights after training
    def save_weights(self, weights_file, sess):
        weights = sess.run(self.parameters)
        keys = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b',
                'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b',
                'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b',
                'conv5_3_W', 'conv5_3_b', 'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b', 'fc9_W', 'fc9_b']
        np.savez(weights_file, **{name: value for name, value in zip(keys, weights)})

# For testing this module
if __name__ == '__main__':

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img_mean = [0,0,0]

    # generate an instance; the instance will be automatically initialized by "__init__( )"
    vgg = vgg16(imgs, img_mean=img_mean, weights='vgg16_weights.npz', sess=sess)
    # If you want to test with batch normalization
    #vgg = vgg16(imgs, img_mean=img_mean, weights='vgg16_weights.npz', sess=sess, bn=True, bn_is_training=True)

    # One should initialize the "FCb" before use
    init_new_vars_op = tf.variables_initializer([vgg.fc4w, vgg.fc4b])               # New; New FC layer needs initialization
    sess.run(init_new_vars_op)                                                      # New; Run initialization
    # If you want to initialize the whole graph, e.g. when testing batch_normalization
    #sess.run(tf.global_variables_initializer())

    # Test input image
    img1 = imread('test.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    print("input image shape:", np.shape(img1))

    # Soft-max classfier
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]                     # Probability

    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

    # Test "save_weights()"
    #vgg.save_weights("weights_out.npz",sess)