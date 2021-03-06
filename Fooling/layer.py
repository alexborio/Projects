import tensorflow as tf
import numpy as np

class DenseLayer(object):
    def __init__(self, n_in, m_out, name, f=tf.nn.relu):
        W = tf.Variable(tf.random_normal(shape=(n_in, m_out)) * 2 / np.sqrt(n_in), name="W_" + name )
        b = tf.Variable(np.zeros(m_out).astype(np.float32), name="b_" + name)
        self.f = f

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.name = name

    def forward(self, X):
        a = tf.matmul(X, self.W) + self.b

        return self.f(a)


class ConvLayer(object):
    def __init__(self, n_in, m_out, filter_sz, stride, name, f=tf.nn.relu):
        W = tf.get_variable(name="W_" +name,
                            shape=(filter_sz, filter_sz, n_in, m_out),
                            initializer=tf.glorot_normal_initializer())

        b = tf.get_variable(name="b_" + name,
                            shape=(m_out,), initializer=tf.zeros_initializer())

        self.W = W
        self.b = b
        self.stride = stride
        self.f = f

        self.params = [self.W, self.b]

    def forward(self, X):

        a = tf.nn.conv2d(X, self.W,
                         [1, self.stride, self.stride, 1],
                         padding='SAME') + self.b

        return self.f(a)

class MaxPoolingLayer(object):
    def __init__(self, filter_sz, stride, name):
        self.filter_sz = filter_sz
        self.stride = stride
        self.name = name
        self.params = []

    def forward(self, X):

        a = tf.layers.max_pooling2d(X, [self.filter_sz, self.filter_sz], strides=self.stride)

        return a

class FSConvLayer(object):
    def __init__(self, n_in, m_out, output_shape, filter_sz, stride, name, f=tf.nn.relu):
        self.filter_sz = filter_sz
        self.name = name

        W = tf.get_variable(name="W_" + name, shape=(filter_sz, filter_sz, m_out, n_in),
                            initializer=tf.glorot_normal_initializer())

        b = tf.get_variable(name="b_"+name,shape=(m_out,),
                            initializer=tf.zeros_initializer())

        self.W = W
        self.b = b
        self.stride = stride
        self.f = f
        self.output_shape = output_shape

    def change_batch_size(self, batch_sz):
        self.output_shape[0] = batch_sz

    def forward(self, X):
        a = tf.nn.conv2d_transpose(
            value=X,
            filter=self.W,
            output_shape=self.output_shape,
            strides=[1, self.stride, self.stride, 1],
        )

        a = tf.nn.bias_add(a, self.b)

        return self.f(a)



