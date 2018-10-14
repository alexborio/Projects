import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer, FSConvLayer
from utilities import _Hook, make_dataset, calculate_accuracy, corrupt_data
import matplotlib.pyplot as plt
import os


class DAE(object):
    def __init__(self):

        self.params_dict = {}
        conv_layers = []
        dense_layers = []

        self.hook = _Hook(self.params_dict, is_training=True)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        conv_layers.append(ConvLayer(1, 8, 3, 1, "input_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "first_pooling_layer"))
        conv_layers.append(ConvLayer(8, 16, 5, 1, "second_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "second_pooling_layer"))
        conv_layers.append(ConvLayer(16, 16, 5, 1, "code"))
        conv_layers.append(FSConvLayer(16, 8, [100, 14, 14, 8], 5, 2, "first_fsconv_layer"))
        conv_layers.append(FSConvLayer(8, 1, [100, 28, 28, 1], 5, 2, "second_fsconv_layer", f=lambda x:x))

        self.conv_layers = conv_layers
        self.dense_layers = dense_layers


    def forward_dae(self, input):

        X = input

        for layer in self.conv_layers:
            X = layer.forward(X)

        logits = X
        reconstruction = tf.nn.sigmoid(X)

        return logits, reconstruction

    def fit(self, train_data, batch_sz):

        train_data[0] = corrupt_data(train_data[0], corruption_level=0.3)

        dataset, capacity = make_dataset(train_data, batch_sz)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_sz)
        dataset = dataset.repeat(5)

        iterator = dataset.make_one_shot_iterator()

        next_examples, next_labels = iterator.get_next()

        logits, reconstruction = self.forward_dae(next_examples)

        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=next_examples, logits=logits))

        self.global_step = tf.train.get_or_create_global_step()

        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)

        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')

        with tf.train.MonitoredTrainingSession( hooks=[self.hook], config=self.config) as sess:
            while not sess.should_stop():
                sess.run(train_op)
                l = sess.run(loss)
                print("Train accuracy: " + str(l))

        #plt.plot(cost_values)
        #plt.show()

    def evaluate(self, test_data, batch_sz):

        test_data[0] = corrupt_data(test_data[0], corruption_level=0.3)
        dataset, capacity = make_dataset(test_data, batch_sz)
        dataset = dataset.batch(batch_sz)

        self.hook.is_training = False
        iterator = dataset.make_one_shot_iterator()
        next_examples, next_labels = iterator.get_next()

        logits, reconstructions = self.forward_dae(next_examples)

        with tf.train.MonitoredTrainingSession(hooks=[self.hook]) as sess:

            while not sess.should_stop():
                X_sim = sess.run(reconstructions)
                im_X_sim = X_sim[0,:,:,:].reshape(28, 28)
                plt.imshow(im_X_sim, cmap='gray')
                plt.show()


train, test = tf.keras.datasets.mnist.load_data()
train_new = []
test_new = []

train_new.append(train[0]/255)
train_new.append(train[1])

test_new.append(test[0]/255)
test_new.append(test[1])

dae = DAE()
dae.fit(train_new, 100)
dae.evaluate(test_new, 100)
