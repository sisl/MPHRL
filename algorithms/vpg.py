import gym
import numpy as np
import ray
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.slim as slim

from algorithms.algorithm import algorithm
from tools.transitional_model import transitional_model
from utils import c, dnn, experience_to_traj, explained_variance, \
    get_returns, threads


class vpg(algorithm):
    def calc_belief_prob_gt(self, s):
        dx_dy = s[13:15]
        correct_model = np.argmax(np.abs(dx_dy))
        belief_prob_gt = np.zeros(c.num_models)
        belief_prob_gt[correct_model] = 1
        return belief_prob_gt

    def get_action(self, s):
        a = self.sess.run(self.sampled_action, feed_dict={
            self.obv_ph: [s]
        })
        return a[0]

    def learn(self, experience):
        trajectories = experience_to_traj(experience)
        returns = get_returns(trajectories)
        time_steps, obvs, acts, rews, next_obvs, dones, internal_states = zip(
            *experience)
        advs = self.calculate_advantage(returns=returns, observations=obvs)
        _, self.bl_loss, self.baseline = self.sess.run(
            [self.train_bl_op, self.bl_loss_tr, self.bl_tr],
            feed_dict={
                self.obv_ph: obvs,
                self.bl_target_ph: returns
            })

        self.next_s_probs = self.calculate_next_s_probs(experience)

        _, self.belief_loss = self.sess.run(
            [self.train_belief_op, self.belief_loss_tr], feed_dict={
                self.obv_ph: obvs,
                self.next_s_probs_ph: self.next_s_probs
            })

        self.fd = {
            self.obv_ph: obvs,
            self.act_ph: acts,
            self.adv_ph: advs,
            self.bl_target_ph: returns,
            self.next_s_probs_ph: self.next_s_probs
        }

        _, self.policy_loss, belief_prob, mean1, std1 = self.sess.run(
            [self.train_policy_op, self.pi_loss_tr,
             self.belief_prob_target, self.weighted_action_means,
             self.action_std], feed_dict=self.fd)
        mean2, std2 = self.sess.run([self.weighted_action_means,
                                     self.action_std], feed_dict=self.fd)
        assert np.allclose(np.sum(belief_prob, axis=1), 1)

        if self.t % c.update_master_interval == 0:
            print('Updating Belief Network...')
            self.sess.run(self.assign_raw_belief_op)
            print('Updated')

        self.results['baseline_losses'][self.t] = self.bl_loss
        self.results['policy_losses'][self.t] = self.policy_loss
        self.results['belief_losses'][self.t] = self.belief_loss
        self.results['explained_variance'][self.t] = explained_variance(
            self.baseline, returns)
        self.results['policy_kl_divergence'][self.t] = self.sess.run(
            self.kl_divergence, feed_dict={
                self.weighted_action_means_ph1: mean1,
                self.weighted_action_means_ph2: mean2,
                self.action_std_ph1: std1,
                self.action_std_ph2: std2
            })

    def update_target_belief_op(self):
        raw_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='raw')
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='target')
        assign_ops = []
        for i in range(len(raw_vars)):
            raw_var = raw_vars[i]
            target_var = target_vars[i]
            assign_ops.append(tf.assign(target_var, raw_var))

        self.assign_raw_belief_op = tf.group(assign_ops)

    def calculate_next_s_prob(self, experience, id):
        probs = []
        for i in range(len(experience)):
            prob = []
            sample = experience[i]
            time_step, obv, act, rew, next_obv, done, internal_s = sample[:7]
            for predictor in self.transitional_predictors:
                next_s_pred = predictor.step(act, obv, internal_s)[0]
                prob.append((next_s_pred, next_obv))
            probs.append(np.array(prob))
            print(id, i, end='\r')
        return np.array(probs)

    def calculate_next_s_probs(self, experience):
        bs = c.bs_per_cpu
        ret = ray.get(
            [self.actors[i].calculate_next_s_pred.remote(
                experience[i * bs:(i + 1) * bs], i) for i in
                range(c.num_cpus)])
        ret = [i for i in ret if len(i.shape) > 1]
        probs = np.concatenate(ret, axis=0)

        ret = np.array(threads(num_threads=c.num_cpus,
                               func=func_calculate_next_s_probs,
                               iterable=zip(probs,
                                                range(c.bs_per_cpu))))
        return ret

    def add_placeholders_op(self):
        self.obv_ph = tf.placeholder(tf.float32, shape=[None, self.obs_dim],
                                     name="obv_ph")
        if self.discrete:
            self.act_ph = tf.placeholder(tf.int64, shape=None, name='sub_act_ph')
        else:
            self.act_ph = tf.placeholder(tf.float32, shape=[None, self.act_dim],
                                         name='sub_act_ph')
        self.adv_ph = tf.placeholder(tf.float32, shape=None, name='sub_adv_ph')
        self.next_s_probs_ph = tf.placeholder(tf.float32, shape=[None,
                                                                 self.num_models],
                                              name='next_s_probs_ph')
        self.bl_target_ph = tf.placeholder(tf.float32, shape=None,
                                           name='bl_target_ph')
        self.belief_prob_gt_ph = tf.placeholder(tf.float32,
                                                shape=[None, c.num_models],
                                                name='belief_prob_gt_ph')
        self.weighted_action_means_ph1 = tf.placeholder(tf.float32, shape=[None,
                                                                           self.act_dim],
                                                        name='weighted_action_means_ph1')
        self.action_std_ph1 = tf.placeholder(tf.float32, shape=[self.act_dim],
                                             name='action_std_ph1')
        self.weighted_action_means_ph2 = tf.placeholder(tf.float32, shape=[None,
                                                                           self.act_dim],
                                                        name='weighted_action_means_ph2')
        self.action_std_ph2 = tf.placeholder(tf.float32, shape=[self.act_dim],
                                             name='action_std_ph2')

    def add_baseline_op(self, scope="baseline"):
        self.bl_tr = tf.squeeze(
            dnn(input=self.obv_ph, output_size=1,
                scope=scope, n_layers=c.pg_bl_n_layers,
                size=c.pg_bl_hidden_fc_size))

    def calculate_advantage(self, returns, observations):
        baseline = self.sess.run(self.bl_tr, feed_dict={
            self.obv_ph: observations
        })
        returns -= baseline

        returns = (returns - np.mean(returns)) / np.std(returns)
        return returns

    def custom_init_op(self):
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obs_dim = 1 if self.discrete else self.env.observation_space.shape[
            0]
        self.act_dim = self.env.action_space.n if self.discrete else \
            self.env.action_space.shape[0]

        self.transitional_predictors = [transitional_model(i) for i in
                                        range(self.num_models)]
        self.add_placeholders_op()
        self.build_model_belief_op()
        self.build_policy_network_op(scope='curr_pi')
        self.add_baseline_op()
        if self.main:
            self.update_target_belief_op()
            self.add_loss_op()
            self.add_optimizer_op()
            self.tensorboard_keys += ['baseline_losses', 'policy_losses',
                                      'belief_losses', 'explained_variance',
                                      'algo_memory', 'gc_collect', 'memory']

    def build_policy_network_op(self, scope="policy"):
        action_means = []
        with tf.variable_scope(scope):
            for i in range(self.num_models):
                with tf.variable_scope('policy_%s' % i):
                    if self.discrete:
                        raise NotImplementedError
                    else:
                        action_means.append(dnn(input=self.obv_ph,
                                                output_size=self.act_dim,
                                                scope='dnn',
                                                n_layers=c.pg_pi_n_layers,
                                                size=c.pg_pi_hidden_fc_size))
        action_mean_tr = tf.stack(action_means, axis=1)
        self.weighted_action_means = tf.reduce_sum(
            tf.expand_dims(self.belief_prob_target, axis=2) * action_mean_tr,
            axis=1)
        act_log_std = tf.get_variable('act_log_std', shape=[self.act_dim],
                                      trainable=True)
        self.action_std = tf.exp(act_log_std)
        multivariate = tfd.MultivariateNormalDiag(
            loc=self.weighted_action_means,
            scale_diag=self.action_std)
        self.sampled_action = tf.random_normal(
            [self.act_dim]) * self.action_std + self.weighted_action_means

        self.logprob = multivariate.log_prob(self.act_ph)

        multivariate1 = tfd.MultivariateNormalDiag(
            loc=self.weighted_action_means_ph1, scale_diag=self.action_std_ph1)

        multivariate2 = tfd.MultivariateNormalDiag(
            loc=self.weighted_action_means_ph2, scale_diag=self.action_std_ph2)
        self.kl_divergence = tf.reduce_mean(
            tf.distributions.kl_divergence(multivariate1, multivariate2))

    def add_loss_op(self):
        self.belief_loss_tr = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(self.belief_target),
                logits=self.belief_logits))
        self.pi_loss_tr = -tf.reduce_mean(self.logprob * self.adv_ph)
        self.bl_loss_tr = tf.losses.mean_squared_error(labels=self.bl_target_ph,
                                                       predictions=self.bl_tr)

    def add_optimizer_op(self):
        bl_opt = tf.train.AdamOptimizer(learning_rate=c.pg_bl_lr)
        self.train_bl_op = slim.learning.create_train_op(self.bl_loss_tr,
                                                         bl_opt,
                                                         summarize_gradients=True)
        policy_opt = tf.train.AdamOptimizer(learning_rate=c.pg_sub_lr)
        self.train_policy_op = slim.learning.create_train_op(self.pi_loss_tr,
                                                             policy_opt,
                                                             summarize_gradients=True)
        belief_opt = tf.train.AdamOptimizer(learning_rate=c.pg_master_ce_lr_source)
        self.train_belief_op = slim.learning.create_train_op(
            self.belief_loss_tr, belief_opt, summarize_gradients=True)

    def build_model_belief_op(self, scope='belief'):
        if self.main:
            with tf.variable_scope('raw'):
                self.belief_logits = dnn(input=self.obv_ph,
                                         output_size=self.num_models,
                                         scope=scope,
                                         n_layers=c.pg_master_n_layers,
                                         size=c.pg_master_hidden_fc_size)
                self.belief_prob = tf.one_hot(
                    indices=tf.argmax(self.belief_logits, axis=1),
                    depth=c.num_models)
        with tf.variable_scope('target'):
            self.belief_logits_target = dnn(input=self.obv_ph,
                                            output_size=self.num_models,
                                            scope=scope,
                                            n_layers=c.pg_master_n_layers,
                                            size=c.pg_master_hidden_fc_size,
                                            trainable=False)
            self.belief_prob_target = tf.one_hot(
                indices=tf.argmax(self.belief_logits_target, axis=1),
                depth=c.num_models)
        # self.belief_prob = tf.nn.softmax(logits=self.belief_logits, axis=1)
        belief_target_joint_prob = self.belief_prob_target * \
                                   self.next_s_probs_ph + 1e-7
        self.belief_target = belief_target_joint_prob / tf.reduce_sum(
            belief_target_joint_prob, axis=1, keepdims=True)
