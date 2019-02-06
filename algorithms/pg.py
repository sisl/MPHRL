import gym
import tensorflow.contrib.distributions as tfd

from utils import *
from algorithms.algorithm import algorithm


class pg(algorithm):
    def get_action(self, s):
        return self.sess.run(self.sampled_action, feed_dict={
            self.obs_ph: np.array([s])
        })[0]

    def learn(self, experience):
        trajectories = experience_to_traj(experience)
        returns = get_returns(trajectories)
        observations = np.array([sample[0] for sample in experience])
        actions = np.array([sample[1] for sample in experience])
        advantages = self.calculate_advantage(returns=returns,
                                              observations=observations)
        baseline_loss = self.update_baseline(returns=returns,
                                             observations=observations)

        self.results['baseline_losses'][self.t] = baseline_loss
        _, main_loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.obs_ph: observations,
            self.action_placeholder: actions,
            self.advantage_placeholder: advantages
        })
        self.results['main_losses'][self.t] = main_loss

    def add_placeholders_op(self):
        self.obs_ph = tf.placeholder(tf.float32, shape=[None,
                                                        self.obs_dim])
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int64, shape=None)
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                        self.action_dim])
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=None)

    def add_baseline_op(self, scope="baseline"):
        self.baseline = tf.squeeze(
            dnn(input=self.obs_ph, output_size=1,
                scope=scope, n_layers=c.pg_bl_n_layers,
                size=c.pg_bl_hidden_fc_size))

        self.baseline_target_ph = tf.placeholder(tf.float32,
                                                 shape=None)
        self.baseline_loss_ph = tf.losses.mean_squared_error(
            self.baseline_target_ph, self.baseline, scope=scope)
        self.baseline_opt = tf.train.AdamOptimizer(
            learning_rate=c.pg_bl_lr)
        self.update_baseline_op = self.baseline_opt.minimize(
            self.baseline_loss_ph)

    def calculate_advantage(self, returns, observations):
        baseline = self.sess.run(self.baseline, feed_dict={
            self.obs_ph: observations
        })
        returns -= baseline

        returns = (returns - np.mean(returns)) / np.std(returns)
        return returns

    def custom_init_op(self):
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obs_dim = 1 if self.discrete else self.env.observation_space.shape[
            0]
        self.action_dim = self.env.action_space.n if self.discrete else \
            self.env.action_space.shape[0]

        self.add_placeholders_op()
        self.build_policy_network_op()
        self.add_loss_op()
        self.add_optimizer_op()
        self.add_baseline_op()
        self.tensorboard_keys.append('baseline_losses')
        self.tensorboard_keys.append('main_losses')

    def update_baseline(self, returns, observations):
        _, self.baseline_loss = self.sess.run([self.update_baseline_op,
                                               self.baseline_loss_ph],
                                              feed_dict={
                                                  self.obs_ph: observations,
                                                  self.baseline_target_ph:
                                                      returns
                                              })
        return self.baseline_loss

    def build_policy_network_op(self, scope="policy_network"):
        if self.discrete:
            action_logits = dnn(input=self.obs_ph,
                                output_size=self.action_dim,
                                scope=scope,
                                n_layers=c.pg_pi_n_layers,
                                size=c.pg_pi_hidden_fc_size)
            self.sampled_action = tf.squeeze(tf.multinomial(action_logits, 1),
                                             axis=1)
            self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.action_placeholder, logits=action_logits)
        else:
            action_means = dnn(input=self.obs_ph,
                               output_size=self.action_dim,
                               scope=scope,
                               n_layers=c.pg_pi_n_layers,
                               size=c.pg_pi_hidden_fc_size)
            log_std = tf.get_variable('log_std', shape=[self.action_dim],
                                      trainable=True)
            action_std = tf.exp(log_std)
            multivariate = tfd.MultivariateNormalDiag(loc=action_means,
                                                      scale_diag=action_std)
            self.sampled_action = tf.random_normal(
                [self.action_dim]) * action_std + action_means
            self.logprob = multivariate.log_prob(self.action_placeholder)

    def add_loss_op(self):
        self.loss = -tf.reduce_mean(self.logprob * self.advantage_placeholder)

    def add_optimizer_op(self):
        self.network_opt = tf.train.AdamOptimizer(learning_rate=c.pg_sub_lr)
        self.train_op = self.network_opt.minimize(self.loss)
