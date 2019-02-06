from __future__ import print_function

import itertools
import random
import traceback
from multiprocessing import Pool

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from matplotlib import pyplot as plt
from config import c


def dnn(input, output_size, scope, n_layers=c.n_layers,
        size=c.hidden_fc_size, output_activation=None):
    out = input
    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = layers.fully_connected(inputs=out, num_outputs=size,
                                         activation_fn=None,
                                         reuse=False)

        out = layers.fully_connected(inputs=out, num_outputs=output_size,
                                     activation_fn=output_activation,
                                     reuse=False)

    return out


def learn():
    # _tr means _tensor
    pool = Pool(c.num_runs)
    ret = pool.map(func, itertools.izip(range(c.num_batches)))
    s = np.reshape(np.array([i[0] for i in ret]),
                   newshape=(-1, env.observation_space.shape[0]))
    a = np.reshape(np.array([i[1] for i in ret]),
                   newshape=(-1, env.action_space.shape[0]))
    next_s = np.reshape(np.array([i[3] for i in ret]),
                        newshape=(-1, env.observation_space.shape[0]))
    r = np.reshape(np.array([i[2] for i in ret]),
                   newshape=(-1))

    s_tr = tf.placeholder(dtype=tf.float32, shape=(None, s.shape[-1]),
                          name='s_tr')
    a_tr = tf.placeholder(dtype=tf.float32, shape=(None, a.shape[-1]),
                          name='a_tr')
    s_a_tr = tf.concat(values=[s_tr, a_tr], axis=1, name='s_a_tr')
    next_s_pred_tr = dnn(input=s_a_tr, output_size=next_s.shape[-1],
                         scope='next_s_pred_tr')
    next_s_gt_tr = tf.placeholder(dtype=tf.float32, shape=(None, s.shape[-1]),
                                  name='next_s_gt_tr')
    l2_loss = tf.losses.mean_squared_error(labels=next_s_gt_tr,
                                           predictions=next_s_pred_tr)
    lr_tr = tf.placeholder(tf.float32, name='lr_tr')
    adam = tf.train.AdamOptimizer(learning_rate=lr_tr)
    train_op = adam.minimize(loss=l2_loss)

    num_samples = s.shape[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = []
    lr = c.base_lr
    lr_l = []
    for i in range(num_samples / c.learn_dynamics_batch_size * c.num_epochs):
        indices = random.sample(range(num_samples), c.learn_dynamics_batch_size)
        loss, _ = sess.run([l2_loss, train_op], feed_dict={
            s_tr: s[indices], a_tr: a[indices], next_s_gt_tr: next_s[indices],
            lr_tr: lr
        })
        losses.append(loss)
        lr_l.append(lr)
        print(loss, num_samples)
        if (i+1) % 100 == 0:
            plt.figure(1)
            plt.plot(range(i+1), losses)
            plt.figure(2)
            plt.plot(range(i+1), lr_l)
            plt.show()
            lr /= 2


def get_experience(trajectory_number):
    experience = []
    s = env.reset()
    iter = 0
    while True:
        a = env.action_space.sample()
        next_s, r, done, info = env.step(a)
        experience.append([s, a, r, next_s])
        if done:
            s = env.reset()
        else:
            s = next_s
        iter += 1
        if iter > c.max_time_steps_per_rollout:
            break
    if trajectory_number % 100 == 0:
        print(trajectory_number)
    return zip(*experience)


def func(_):
    try:
        return get_experience(*_)
    except:
        traceback.print_exc()
        raise Exception


env = gym.make(c.env_name)
learn()
