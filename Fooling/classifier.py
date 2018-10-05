import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer
import random

def randomize_and_batch(train_data, batch_sz):

    indices = tf.random_shuffle(tf.range(len(train_data[0])))
    images_data = tf.reshape(tf.convert_to_tensor(train_data[0]), [-1, 28, 28, 1])
    labels_data = tf.one_hot(tf.convert_to_tensor(train_data[1]), 10)

    images = tf.gather(images_data, indices)
    labels = tf.gather(labels_data, indices)
    images_batch, labels_batch = tf.train.batch([images, labels], batch_size=batch_sz, enqueue_many=True,
                                                shapes=([28, 28, 1], [10, ]))

    return images_batch, labels_batch, images, labels



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

    def fit(self, train_data):

        batch_sz = 100

        images_batch, labels_batch, images, labels = randomize_and_batch(train_data, batch_sz)
        images_batch = tf.cast(images_batch, tf.float32)

        X = images_batch
        for layer in self.conv_layers:
            X = layer.forward(X)

        X = tf.reshape(X, [-1, 7*7*64])

        for layer in self.dense_layers:
            X = layer.forward(X)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_batch, logits=X)

        train_op = tf.train.AdamOptimizer().minimize(loss)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        epochs = 10

        cost_values = []

        for epoch in range(epochs):

            self.sess.run([images, labels]) # shuffle TODO: check that works properly

            try:

                while not self.coord.should_stop():
                    self.sess.run([images_batch, labels_batch]) # compute new batch TODO: check that works properly
                    _, cost_value = self.sess.run((train_op, loss))
                    cost_values.append(cost_value)
                    print(cost_value)

            finally:
                self.coord.request_stop()
                self.coord.join(self.threads)

# mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
# train = mnist.train
train, test = tf.keras.datasets.mnist.load_data()
classifier = Classifier()
classifier.fit(train)



