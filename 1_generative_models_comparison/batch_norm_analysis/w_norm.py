import os
from datetime import datetime as dt
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from util.util import plot

OUT_DIR = 'w_norm/'

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3

# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

xavier = tf.contrib.layers.xavier_initializer()
zeros = tf.zeros_initializer()
Q_W1 = tf.get_variable("Q_W1", [X_dim, h_dim], initializer=xavier)
Q_b1 = tf.get_variable("Q_b1", [h_dim], initializer=zeros)

Q_W2_mu = tf.get_variable("Q_W2_mu", [h_dim, z_dim], initializer=xavier)
Q_b2_mu = tf.get_variable("Q_b2_mu", [z_dim], initializer=zeros)

Q_W2_sigma = tf.get_variable("Q_W2_sigma", [h_dim, z_dim], initializer=xavier)
Q_b2_sigma = tf.get_variable("Q_b2_sigma", [z_dim], initializer=zeros)


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.get_variable("P_W1", [z_dim, h_dim], initializer=xavier)
P_b1 = tf.get_variable("P_b1", [h_dim], initializer=zeros)

P_W2 = tf.get_variable("P_W2", [h_dim, X_dim], initializer=xavier)
P_b2 = tf.get_variable("P_b2", [X_dim], initializer=zeros)


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

z_mu, z_logvar = Q(X)
z_sample = sample_z(z_mu, z_logvar)
_, logits = P(z_sample)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, 1)
# VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if os.path.exists(OUT_DIR):
    files = [f for f in os.listdir(OUT_DIR)]
    for f in files:
        os.remove(os.path.join(OUT_DIR, f))
else:
    os.makedirs(OUT_DIR)

i = 0

log_file = open('{}log.txt'.format(OUT_DIR), 'w+')
cur_time = dt.now()
testSamples = np.random.randn(16, z_dim)
for it in range(20001):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    if it % 1000 == 0:
        for out in [log_file, stdout]:
            print('Iter: {}; Loss: {:.4};'
                  .format(it, loss), file=out, flush=True)

        samples = sess.run(X_samples, feed_dict={z: testSamples})

        fig = plot(samples)
        plt.savefig('{}{}.png'.format(OUT_DIR, str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

for out in [log_file, stdout]:
    print('Time: {} seconds'.format((dt.now() - cur_time).seconds), file=out, flush=True)
