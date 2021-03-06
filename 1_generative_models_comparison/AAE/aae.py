import os
from datetime import datetime as dt
from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from util.util import plot, xavier_init

OUT_DIR = 'out/'

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 10
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3

""" Q(z|X) """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q = [Q_W1, Q_W2, Q_b1, Q_b2]


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z = tf.matmul(h, Q_W2) + Q_b2
    return z


""" P(X|z) """
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_P = [P_W1, P_W2, P_b1, P_b2]


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


""" D(z) """
D_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def D(z):
    h = tf.nn.relu(tf.matmul(z, D_W1) + D_b1)
    logits = tf.matmul(h, D_W2) + D_b2
    prob = tf.nn.sigmoid(logits)
    return prob


""" Training """
z_sample = Q(X)
_, logits = P(z_sample)

# Sample from random z
X_samples, _ = P(z)

# E[log P(X|z)]
recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X))

# Adversarial loss to approx. Q(z|X)
D_real = D(z)
D_fake = D(z_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

AE_solver = tf.train.AdamOptimizer().minimize(recon_loss, var_list=theta_P + theta_Q)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_Q)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if os.path.exists(OUT_DIR):
    files = [f for f in os.listdir(OUT_DIR)]
    for f in files:
        os.remove(os.path.join(OUT_DIR, f))
else:
    os.makedirs(OUT_DIR)

i = 0

log_file = open('out/log.txt', 'w+')
cur_time = dt.now()
for it in range(20001):
    X_mb, _ = mnist.train.next_batch(mb_size)
    z_mb = np.random.randn(mb_size, z_dim)

    _, recon_loss_curr = sess.run([AE_solver, recon_loss], feed_dict={X: X_mb})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, z: z_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb})

    if it % 1000 == 0:
        for out in [log_file, stdout]:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; Recon_loss: {:.4}'
                  .format(it, D_loss_curr, G_loss_curr, recon_loss_curr), file=out, flush=True)

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

for out in [log_file, stdout]:
    print('Time: {} seconds'.format((dt.now() - cur_time).seconds), file=out, flush=True)
