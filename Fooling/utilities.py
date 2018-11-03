import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from layer import DenseLayer, ConvLayer, MaxPoolingLayer
import matplotlib.pyplot as plt
import os

class _Hook(tf.train.SessionRunHook):

    def __init__(self, params_dict, is_training=False):
        self.params_dict = params_dict
        self.assign_ops = [] # list for assignment operations
        self.assignment_performed = False # indicates wether weights have been loaded
        self.is_training = is_training
        self.train_vars = []

    """Append assignment ops to a graph = load trained weights and biases"""
    def begin(self):
        if (len(self.params_dict) > 0):
            graph = tf.get_default_graph()
            variables = graph._collections['trainable_variables']

            for variable in variables:
                if variable.name in self.params_dict:
                    self.assign_ops.append( variable.assign(self.params_dict[variable.name]))

    """Perform assignment operations"""
    def before_run(self, run_context):
        if (len(self.assign_ops) > 0 and not self.assignment_performed):
            for op in self.assign_ops:
                run_context.session.run(op)

            self.assignment_performed = True


    """Save trained params into a dictionary provided"""
    def end(self, session):
        if self.is_training:
            variables = session.graph._collections['trainable_variables']

            for variable in variables:
                self.params_dict.update({variable.name: session.run(variable)})

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

def corrupt_data(data, corruption_level=0.3):
    data = np.array([x * (np.random.uniform(size=(28, 28)) < (1-corruption_level)) for x in data])
    return data