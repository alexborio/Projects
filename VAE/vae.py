
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

encoder_layers = []
decoder_layers = []

Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli

class Layer(object):
    def __init__(self, n, m, f=tf.nn.relu):
        self.W = tf.Variable(tf.random_normal(shape=(n, m))*2 / np.sqrt(n), dtype=tf.float32)
        self.W = tf.cast(self.W, dtype=tf.float32)
        self.c = tf.Variable(tf.zeros(m), dtype=tf.float32)
        self.c = tf.cast(self.c, dtype=tf.float32)
        self.f = f

    def forward(self, X):
        return self.f(tf.matmul(X, self.W) + self.c)


def KLDivergence(mu, sigma):
    KL1 = tf.log(sigma)
    KL2 = (1 + tf.pow((-mu), 2))/2/tf.pow(sigma, 2) - 0.5
    KL = KL1 + KL2
    return tf.reduce_sum(KL, axis=1)


mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
X = mnist.train.images

n_input = 28*28

hidden_layer_sizes = [200, 150, 100, 20, 2]
X_in = tf.placeholder(dtype=tf.float32, shape=(None, n_input))
Z = X_in
n = n_input

for m in hidden_layer_sizes[:-1]:

    layer = Layer(n, m)
    Z = layer.forward(Z)
    encoder_layers.append(layer)
    n = m


m_latent = hidden_layer_sizes[-1] * 2

layer = Layer(n, m_latent, lambda x: x)

Z = layer.forward(Z)
encoder_layers.append(layer)

mu = Z[:, :(m_latent // 2)]
sigma = tf.exp(Z[:, (m_latent // 2):])

E = tf.placeholder(dtype=tf.float32, shape=(None, hidden_layer_sizes[-1]))

n = m_latent // 2

Z = E*sigma + mu

for m in reversed(hidden_layer_sizes[:-1]):
    layer = Layer(n, m)
    Z = layer.forward(Z)
    decoder_layers.append(layer)
    n = m

layer = Layer(n, n_input, lambda x: x)
Z = layer.forward(Z)
decoder_layers.append(layer)

kl = -tf.log(sigma) + 0.5 * (sigma ** 2 + mu ** 2) - 0.5
kl = tf.reduce_sum(kl, axis=1)

#kl = KLDivergence(mu, sigma)

probs = tf.contrib.distributions.Bernoulli(logits=Z).log_prob(X_in)
cost = tf.reduce_sum(tf.reduce_sum(probs, 1) - kl)
train_op = tf.train.RMSPropOptimizer(0.001).minimize(-cost)
sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()

sess.run(init_op)

N = (X.shape)[0]
costs = []
n_batch_sz = 100
epochs = 50
n_batches = N // n_batch_sz
X = (X > 0.5).astype(np.float32)
for epoch in range(epochs):

    np.random.shuffle(X)

    for i in range(n_batches):

        dict1 = {X_in: X[i*n_batch_sz:(i + 1)*n_batch_sz, :]}
        dict2 = {E: np.reshape(np.random.randn(m_latent // 2), (1, m_latent // 2))}
        dict1.update(dict2)
        _, c = sess.run((train_op, cost), feed_dict=dict1)
        c /= n_batch_sz
        costs.append(c)
        print(c)


done = False

Z_in = tf.placeholder(dtype=tf.float32, shape=(None, hidden_layer_sizes[-1]))
Z_sim = Z_in
for layer in decoder_layers:
    Z_sim = layer.forward(Z_sim)

Z_sim_out = tf.nn.sigmoid(Z_sim)

while not done:
    feed = {Z_in: np.reshape(np.random.randn(m_latent // 2), (1, m_latent // 2))}
    X_sim = sess.run(Z_sim_out, feed_dict=feed)

    im_X_sim = X_sim.reshape(28, 28)
    plt.imshow(im_X_sim, cmap='gray')
    plt.show()

    ans = input("Generate another?")
    if ans and ans[0] in ('n' or 'N'):
      done = True