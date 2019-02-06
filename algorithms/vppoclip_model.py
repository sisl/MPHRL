from algorithms.vppo import *
from utils import func_calculate_next_s_probs, threads


class vppoclip_model(vppoclip):
    def custom_init_op(self):
        self.clip_param_tr = c.clip_param
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obs_dim = 1 if self.discrete else self.env.observation_space.shape[
            0]
        self.act_dim = self.env.action_space.n if self.discrete else \
            self.env.action_space.shape[0]

        self.transitional_predictors = [transitional_model(i) for i in
                                        range(self.num_models)]
        self.add_placeholders_op()
        self.add_build_master_op()
        self.build_model_network_op()
        self.add_baseline_op()

        self.pi = self.build_policy_network_op(scope='curr_pi',
                                               trainable=self.main and
                                                         not c.fix_sub)
        if self.main:
            self.old_pi = self.build_policy_network_op(scope='old_pi',
                                                       trainable=False)

            self.add_update_old_subpolicy_op()
            self.add_update_target_master_op()
            self.add_train_op()
            self.tensorboard_keys += ['bl_loss', 'subpolicy_loss',
                                      'master_ce_loss_tr', 'improved_variance',
                                      'gc_collect', 'policy_kl_divergence_tr',
                                      'pg_master_ce_lr', 'sub_lr', 'bl_lr',
                                      'pi_grad_norm_tr', 'bl_grad_norm_tr',
                                      'master_ce_grad_norm_tr', 'timesteps_so_far',
                                      'clip_param', 'master_pg_lr',
                                      'sub_loss_tr', 'model_percentage_tr',
                                      'model_network_loss_tr']
            self.bl_lr = c.pg_bl_lr
            self.pi_lr = c.pg_sub_lr
            self.belief_lr = c.pg_master_ce_lr_source

    def add_train_op(self):
        self.add_master_ce_train_op()
        self.add_subpolicy_train_op()
        self.add_baseline_train_op()
        self.add_master_pg_train_op()
        self.add_model_network_train_op()

    def build_model_network_op(self, scope="models", trainable=True):
        self.model_pred = []
        self.obv_and_ac = tf.concat([self.obv_ph, self.act_ph], axis=1)
        with tf.variable_scope(scope):
            for i in range(self.num_models):
                with tf.variable_scope('model_%s' % i):
                    self.model_pred.append(dnn(input=self.obv_and_ac,
                                               output_size=self.obs_dim,
                                               scope=scope,
                                               n_layers=2,
                                               size=64,
                                               trainable=trainable))
        self.model_pred_tr = tf.stack(self.model_pred, axis=1)

    def add_model_network_train_op(self):
        # NOTE: OPTIMIZER
        model_opt = tf.train.AdamOptimizer(learning_rate=3e-4, epsilon=1e-5)
        # NOTE: LOSS
        self.model_losses_list = []
        for n_obv in self.model_pred:
            self.model_losses_list.append(
                tf.reduce_mean(tf.square(n_obv - self.n_obv_ph), axis=1))
        self.model_losses_tr = tf.stack(self.model_losses_list, axis=1)
        self.model_selection = tf.argmin(self.model_losses_tr, axis=1)
        self.model_percentage_tr = tf.reduce_mean(
            tf.cast(self.model_selection, dtype=tf.float64))
        self.smallest_loss_one_hot = tf.one_hot(self.model_selection,
                                                depth=c.num_models)
        self.model_network_loss_tr = tf.reduce_mean(
            self.model_losses_tr * self.smallest_loss_one_hot)
        self.train_model_network_op = slim.learning.create_train_op(
            self.model_network_loss_tr, optimizer=model_opt,
            clip_gradient_norm=c.gradclip, summarize_gradients=True)

    def calc_post_belief(self, obvs, acts, next_obvs):
        next_obv_pred = self.sess.run(self.model_pred_tr,
                                      feed_dict={self.obv_ph_raw: obvs,
                                                 self.act_ph: acts})

        ret = np.array(threads(num_threads=c.num_cores,
                               func=func_calculate_next_s_probs,
                               iterable=zip(next_obv_pred, range(
                                       c.bs_per_cpu * c.num_cpus))))

        assert ret.shape[0] == c.num_cpus * c.bs_per_cpu
        return ret

    def learn(self, experience):
        time_steps, obvs, acts, rews, next_obvs, dones, internal_states = zip(
            *experience)

        obvs = np.array(obvs)
        next_obvs = np.array(next_obvs)
        acts = np.array(acts)
        rews = np.array(rews)
        dones = list(dones)

        # NOTE: BASELINE PRE
        baseline = self.sess.run(self.bl_tr, feed_dict={
            self.obv_ph_raw: obvs
        })
        # NOTE: ADVANGTAGES
        seg = generate_seg(dones, rews, baseline)
        add_vtarg_and_adv(seg)
        advs, vtarg = seg["adv"], seg["tdlamret"]
        advs = (advs - advs.mean()) / (advs.std() + 1e-7)

        self.ob_rms.update(obvs, self.sess)
        # NOTE: UPDATE OLD POLICY
        self.sess.run(self.assign_old_sub_op)
        # NOTE: NEXT S PROBABILITIES
        self.next_s_probs = self.calc_post_belief(obvs, acts, next_obvs)
        num_time_steps = obvs.shape[0]

        assert num_time_steps == self.next_s_probs.shape[0], (
            num_time_steps, self.next_s_probs.shape[0])

        # NOTE: SHUFFLING
        bs = c.mini_bs
        num_iter = int(num_time_steps // bs)
        order = np.random.permutation(num_time_steps).tolist()
        obvs = obvs[order]
        next_obvs = next_obvs[order]
        acts = acts[order]
        advs = advs[order]
        vtarg = vtarg[order]
        self.next_s_probs = self.next_s_probs[order]

        self.fd = {
            self.obv_ph_raw: obvs,
            self.act_ph: acts,
            self.adv_ph: advs,
            self.n_obv_ph: next_obvs,
            self.bl_target_ph: vtarg,
            self.next_s_probs_ph: self.next_s_probs,
            self.sub_lr_ph: self.pi_lr,
            self.lr_mult_ph: self.lr_mult,
            self.master_pg_lr_ph: self.master_pg_lr,
            self.master_ce_lr_ph: self.belief_lr
        }

        # NOTE: TRAIN BELIEF NETWORK
        _, __ = self.sess.run(
            [self.train_master_ce_op, self.train_master_pg_op],
            feed_dict=self.fd)

        self.baseline = []
        self.policy_loss = []
        for i in range(c.optim_epochs):
            for j in range(num_iter):
                # NOTE: Update Baseline Loss
                sample_fd = {
                    self.obv_ph_raw: obvs[j * bs: (j + 1) * bs],
                    self.act_ph: acts[j * bs: (j + 1) * bs],
                    self.adv_ph: advs[j * bs: (j + 1) * bs],
                    self.bl_target_ph: vtarg[j * bs: (j + 1) * bs],
                    self.sub_lr_ph: self.pi_lr,
                    self.lr_mult_ph: self.lr_mult,
                    self.bl_lr_ph: self.bl_lr
                }

                _, __, baseline, self.bl_loss, policy_loss = self.sess.run(
                    [self.train_bl_op, self.train_sub_op, self.bl_tr,
                     self.bl_loss_tr, self.sub_surr_loss_tr],
                    feed_dict=sample_fd)

                if isinstance(baseline, np.float32):
                    baseline = np.array([baseline])
                if i == c.optim_epochs - 1 and baseline != []:
                    self.baseline.append(baseline)
                    self.policy_loss.append(policy_loss)
                    if c.validate:
                        belief_prob = self.sess.run(self.master_prob_old,
                                                    feed_dict=sample_fd)
                        assert np.allclose(np.sum(belief_prob, axis=1), 1)

        # NOTE: TRAIN MODEL NETWORK
        _ = self.sess.run(self.train_model_network_op, feed_dict=self.fd)

        # NOTE: DONE
        full_bl = np.concatenate(self.baseline, axis=0)
        self.improved_variance = explained_variance(
            np.concatenate(self.baseline, axis=0), vtarg[:full_bl.shape[0]])
        self.policy_loss = np.mean(self.policy_loss)
        self.clip_param = self.sess.run(self.clip_param_tr,
                                        feed_dict={
                                            self.lr_mult_ph: self.lr_mult
                                        })
        self.sess.run(self.assign_old_sub_op)
        if (self.t + 1) % c.update_master_interval == 0:
            print('Updating Belief Network...')
            self.sess.run(self.update_target_master_op)
            print('Updated')
