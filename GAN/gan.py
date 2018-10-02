import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

class DenseLayer(object):
    def __init__(self, n_in, m_out, name, f=tf.nn.relu):
        W = tf.Variable(tf.random_normal(shape=(n_in, m_out)) * 2 / np.sqrt(n_in), name=name + "_W")
        b = tf.Variable(np.zeros(m_out).astype(np.float32), name=name + "_b")
        self.f = f

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.name = name

    def forward(self, X):
        a = tf.matmul(X, self.W) + self.b

        return self.f(a)


def plot_grid(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    i = 0

    for sample in samples:
        if i > 15:
            break
        ax = plt.subplot(gs[i])
        plt.axis('off')
        i += 1

        ax.set_aspect('equal')

        plt.imshow(sample.reshape(28, 28), cmap='gray' )

    return fig


class GAN(object):
    def __init__(self, n_latent, d_sizes, g_sizes, index):

        self.index = index
        img_size = 28
        self.img_size = img_size
        self.d_sizes = d_sizes
        self.g_sizes = g_sizes
        self.g_layers = []
        self.d_layers = []

        self.n_latent = n_latent

        self.g_vars = []
        self.d_vars = []

        n_in = img_size**2

        self.g_sizes.append(img_size**2)
        self.d_sizes.append(1)
        mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
        self.X = mnist.train.images
        count = 0

        for d_size in self.d_sizes:
            name = "discriminator_" + str(count)
            if d_size != self.d_sizes[-1]:
                layer = DenseLayer(n_in, d_size, name)
            else:
                layer = DenseLayer(n_in, d_size, name, lambda x:x)

            self.d_layers.append(layer)
            self.d_vars.append(layer.params)
            count += 1
            n_in = d_size

        n_in = n_latent
        count = 0
        for g_size in self.g_sizes:
            name = "generator_" + str(count)

            if g_size != self.g_sizes[-1]:
                layer = DenseLayer(n_in, g_size, name)
            else:
                layer = DenseLayer(n_in, g_size, name, tf.nn.sigmoid)

            self.g_layers.append(layer)
            self.g_vars.append(layer.params)
            count += 1
            n_in = g_size

    def sample_generator(self, noise):
        # E = tf.placeholder(tf.float32, shape=(examples, self.n_latent))
        sample = noise
        for layer in self.g_layers:
            sample = layer.forward(sample)

        return sample

    def forward_discriminator(self, X):
        out = X
        for layer in self.d_layers:
            out = layer.forward(out)

        return out


    def fit(self):

        E = tf.placeholder(dtype = tf.float32, shape=(None, self.n_latent))
        Z = tf.placeholder(dtype=tf.float32, shape=(None, self.img_size**2))

        sampled_images = self.sample_generator(E)
        logits = self.forward_discriminator(Z)
        logits_sampled = self.forward_discriminator(sampled_images)
        sampled_images_test = self.sample_generator(E)

        d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_sampled, labels=tf.zeros_like(logits_sampled))

        g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_sampled, labels=tf.ones_like(logits_sampled)))
        d_cost = tf.reduce_mean(d_cost_fake) + tf.reduce_mean(d_cost_real)

        d_train_op = tf.train.AdamOptimizer().minimize(d_cost, var_list=self.d_vars)
        g_train_op = tf.train.AdamOptimizer().minimize(g_cost, var_list=self.g_vars)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

        d_costs = []
        g_costs = []
        batch_sz = 16
        epochs = 1000

        n_batches = len(self.X) // batch_sz

        if not os.path.exists('../samples' + str(self.index) +'/'):
            os.makedirs('../samples' + str(self.index) +'/')

        k = 0
        for i in range(epochs):
            print("epoch:", i)
            np.random.shuffle(self.X)
            for j in range(n_batches):
                batch = self.X[j * batch_sz:(j + 1) * batch_sz]
                W = np.random.uniform(-1, 1, size=(batch_sz, self.n_latent))
                _, d_cost_value = self.sess.run((d_train_op, d_cost), feed_dict={Z: batch, E: W})
                _, g_cost_value = self.sess.run((g_train_op, g_cost), feed_dict={E: W})
                d_costs.append(d_cost_value)
                g_costs.append(g_cost_value)

                if j % 100 == 0:
                    print("epoch " + str(i) + " iter " + str(j) + " d cost: " + str(d_costs[-1]) + " g cost: " + str(g_costs[-1]))
                    Z_out_test = self.sess.run(tf.nn.sigmoid(sampled_images_test),
                                               feed_dict={E: W})

                    fig = plot_grid(Z_out_test)
                    plt.savefig('../samples' + str(self.index) +'/' + str(k)+ '.png')
                    k += 1
                    plt.close(fig)




g = GAN(50, [150], [300], 1)
g.fit()

g2 = GAN(30, [150, 500], [150, 500], 2)
g2.fit()








