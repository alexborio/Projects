import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer


class Classifier(object):
    def __init__(self):

        conv_layers = []
        dense_layers = []

        conv_layers.append(ConvLayer(1, 32, 3, 1, "input_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "first_pooling_layer"))
        conv_layers.append(ConvLayer(32, 64, 5, 1, "second_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "second_pooling_layer"))
        dense_layers.append(DenseLayer(7*7*64, 1024, "first_dense_layer") ) #size halved by a max pool layer and then halved again by the second max pool layer
        dense_layers.append(DenseLayer(1024, 10, "logits_layer", lambda x:x))  #logits layer no nonlinearity

        self.conv_layers = conv_layers
        self.dense_layers = dense_layers


    def fit(self, X_train, Y_train):

        self.X_train = X_train
        self.Y_train = Y_train

        X_in = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        labels = tf.placeholder(tf.float32, shape=[None, 10])

        X = X_in
        for layer in self.conv_layers:
            X = layer.forward(X)

        X = tf.reshape(X, [-1, 7*7*64])

        for layer in self.dense_layers:
            X = layer.forward(X)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=X)

        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

        batch_sz = 64
        epochs = 10

        cost_values = []

        for epoch in range(epochs):
            _, cost_value = self.sess.run((train_op, loss), feed_dict={labels: self.Y_train, X_in: self.X_train})
            cost_values.append(cost_value)
            print(cost_value)


mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
X_train = mnist.train.images
X_train = np.reshape(X_train, [-1, 28, 28, 1])
Y_train = mnist.train.labels
classifier = Classifier()
classifier.fit(X_train[:1000, :, :, :], Y_train[:1000, :])



