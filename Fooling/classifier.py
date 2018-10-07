import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer
import matplotlib.pyplot as plt
import os

def make_dataset(train_data, batch_sz):

    n_data = len(train_data[0])
    images_data = np.reshape(train_data[0], [-1, 28, 28, 1]).astype(np.float32)

    nb_classes = 10
    labels_data = (np.eye(nb_classes)[train_data[1]]).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((images_data, labels_data))
    capacity = n_data // batch_sz

    return dataset, capacity

def calculate_accuracy(predictions, labels):

    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            , dtype=tf.float32)
    )

    return accuracy



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

    def forward_classifier(self, input):

        X = input

        for layer in self.conv_layers:
            X = layer.forward(X)

        X = tf.reshape(X, [-1, 7*7*64])

        for layer in self.dense_layers:
            X = layer.forward(X)

        logits = X
        predicted_classes = tf.nn.softmax(X)

        return logits, predicted_classes



    def fit(self, train_data, batch_sz):

        dataset, capacity = make_dataset(train_data, batch_sz)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(batch_sz)
        dataset = dataset.repeat(5)

        iterator = dataset.make_one_shot_iterator()

        next_examples, next_labels = iterator.get_next()

        logits, predicted_classes = self.forward_classifier(next_examples)

        accuracy = calculate_accuracy(predicted_classes, next_labels)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=next_labels, logits=logits)

        self.global_step = tf.train.get_or_create_global_step()

        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)

        cost_values = []
        saver = tf.train.Saver()

        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')

        with tf.train.MonitoredTrainingSession(checkpoint_dir="tmp/") as sess:
            while not sess.should_stop():
                sess.run(train_op)
                acc = sess.run(accuracy)

        #plt.plot(cost_values)
        #plt.show()

    def evaluate(self, test_data, batch_sz):

        dataset, capacity = make_dataset(test_data, batch_sz)
        dataset = dataset.batch(batch_sz)

        iterator = dataset.make_one_shot_iterator()
        next_examples, next_labels = iterator.get_next()

        logits, predicted_classes = self.forward_classifier(next_examples)
        accuracy = calculate_accuracy(predicted_classes, next_labels)

        with tf.train.MonitoredTrainingSession(checkpoint_dir="tmp/") as sess:

            while not sess.should_stop():
                acc = sess.run(accuracy)
                print(acc)

    def print_weights(self):

        print("-------------------Printing weights ------------------")
        with tf.Session() as sess:
            i = 0
            for layer in self.conv_layers:
                params = sess.run(layer.params)
                print(params)
                i+=1

            i = 0
            for layer in self.dense_layers:
                params = sess.run(layer.params)
                print(params)
                i+=1



train, test = tf.keras.datasets.mnist.load_data()

classifier = Classifier()
classifier.fit(train, 100)
# classifier.print_weights()
classifier.evaluate(test, 10000)
# classifier.print_weights()



