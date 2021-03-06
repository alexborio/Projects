import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer
from utilities import _Hook, make_dataset, calculate_accuracy
import matplotlib.pyplot as plt
import os


class Classifier(object):
    def __init__(self):

        self.params_dict = {}
        conv_layers = []
        dense_layers = []

        self.hook = _Hook(self.params_dict, is_training=True)


        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        conv_layers.append(ConvLayer(1, 32, 3, 1, "input_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "first_pooling_layer"))
        conv_layers.append(ConvLayer(32, 64, 5, 1, "second_conv_layer"))
        conv_layers.append(MaxPoolingLayer(2, 2, "second_pooling_layer"))
        dense_layers.append(DenseLayer(7*7*64, 1024, "first_dense_layer") ) #size halved by a max pool layer and then halved again by the second max pool layer
        dense_layers.append(DenseLayer(1024, 10, "logits_layer", lambda x:x))  #logits layer no nonlinearity

        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        self.fooling_input = tf.get_variable(name="fooling_number", shape=(1, 28, 28, 1), initializer=tf.glorot_normal_initializer())
        self.fooling_logits, self.fooling_predicted_classes = self.forward_classifier(self.fooling_input)
        self.fooling_label = tf.placeholder(dtype=tf.float32, shape=(1, 10))
        self.fooling_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.fooling_label, logits=self.fooling_logits)
        self.fooling_train_op = tf.train.AdamOptimizer().minimize(self.fooling_loss, var_list=[self.fooling_input])


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

        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')

        with tf.train.MonitoredTrainingSession( hooks=[self.hook], config=self.config) as sess:
            while not sess.should_stop():
                sess.run(train_op)
                acc = sess.run(accuracy)
                print("Train accuracy: " + str(acc))

        #plt.plot(cost_values)
        #plt.show()

    def evaluate(self, test_data, batch_sz):

        dataset, capacity = make_dataset(test_data, batch_sz)
        dataset = dataset.batch(batch_sz)

        self.hook.is_training = False
        iterator = dataset.make_one_shot_iterator()
        next_examples, next_labels = iterator.get_next()

        logits, predicted_classes = self.forward_classifier(next_examples)
        accuracy = calculate_accuracy(predicted_classes, next_labels)

        with tf.train.MonitoredTrainingSession(hooks=[self.hook]) as sess:

            while not sess.should_stop():
                acc = sess.run(accuracy)
                print("Test accuracy: " + str(acc))


    def fool(self, number):

        fooling_label = np.zeros((1, 10))
        fooling_label[0][number] = 0.999

        epochs = 1000
        images = []
        self.hook.is_training = True

        with tf.train.MonitoredTrainingSession(hooks=[self.hook], config=self.config) as sess:
            for epoch in range(epochs):
                sess.run(self.fooling_train_op, feed_dict={self.fooling_label: fooling_label})
                pred_class_value = sess.run(self.fooling_input)
                image = pred_class_value.reshape(28, 28)
                images.append(image)
                print(sess.run(self.fooling_predicted_classes))

            plt.imshow(image, cmap='gray')
            plt.show()

        self.hook.assignment_performed = False
        self.hook.is_training = False


train, test = tf.keras.datasets.mnist.load_data()

train_new = []
test_new = []

train_new.append(train[0]/255)
train_new.append(train[1])

test_new.append(test[0]/255)
test_new.append(test[1])



classifier = Classifier()

classifier.fit(train_new, 100)
classifier.fool(0)
classifier.fool(9)
classifier.evaluate(test_new, 100)



