import math
import socket

import gym
import ray
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.slim as slim

from algorithms.algorithm import algorithm
from utils import *


class vppo(algorithm):
    def reinit(self, task_id):
        self.task_id = task_id
        self.env = gym.make('%s%s' % (task_id, c.env_name))
        self.env.seed(self.run_id)
        self.timesteps_used_list.append(self.timesteps_so_far / 1e6)
        self.timesteps_so_far = float(c.prev_timestep)
        self.max_reward = -math.inf
        self.max_eval_suc_mov = -math.inf
        self.sess.run(self.reset_obfilter_op)
        if not c.transfer:
            self.sess.run(self.init_op)
        if c.reset:
            self.sess.run(self.reset_non_sub_op)
        if c.restore_model and c.always_restore:
            self.restore_model_best_op()

    def get_action(self, s, view_master=False, sub=None, pos=None):
        if sub is not None:
            a = self.sess.run(self.curr_pi['sampled_action_sub_%s' % sub], feed_dict={self.obv_ph_raw: [s]})
            belief = np.zeros((1, c.num_models))
            belief[0, sub] = 1
        else:
            if c.oracle_master:
                corridors = self.env.env.corridors
                sub = corridors[corridor_number(pos[0], pos[1], corridors)]['model']
                a, belief = self.sess.run([self.curr_pi['sampled_action'], self.master_prob],
                                          feed_dict={
                                              self.obv_ph_raw: [s],
                                              self.oracle_m_label_ph: [sub],
                                              self.oracle_master_ph: self.oracle_or_not()
                                          })
            else:
                a, belief = self.sess.run([self.curr_pi['sampled_action'], self.master_prob],
                                          feed_dict={self.obv_ph_raw: [s]})
                if c.model_primitive_pg:
                    mp_pred = self.sess.run(self.curr_mp_pi['sampled_action'],
                                            feed_dict={self.obv_ph_raw: [s], self.sub_act_ph: a})
                    return np.squeeze(a), None

        if view_master:
            return np.squeeze(a), belief[0]
        else:
            return np.squeeze(a), None

    def oracle_or_not(self):
        return [float(self.t >= c.oracle_turn_on and c.oracle_master)]

    def calc_velo(self, info):
        if self.is_mujoco():
            velocity = np.array([i['velocity'] for i in info])
            self.dx_mean, self.dy_mean = np.mean(velocity, axis=0)
            self.dx_std, self.dy_std = np.std(velocity, axis=0)
        else:
            self.dx_mean, self.dy_mean, self.dx_std, self.dy_std = 0, 0, 0, 0

    def learn(self, experience):
        _, sub_obvs, sub_acts, sub_rews, sub_n_obvs, sub_dones, sub_infos = zip(*experience)

        self.avg_sub_env_index = np.mean([i['sub_env_index'] for i in sub_infos]) if self.is_mujoco() else 0
        self.distance_to_goal = np.mean([sub_infos[i]['dist'] for i in range(len(sub_infos)) if sub_dones[i]])
        self.survival_reward = np.mean([i['reward_survive'] for i in sub_infos])
        self.forward_reward = np.mean([i['reward_forward'] for i in sub_infos])
        self.control_reward = np.mean([i['reward_ctrl'] for i in sub_infos])
        self.contact_reward = np.mean([i['reward_contact'] for i in sub_infos])
        self.calc_velo(sub_infos)

        sub_obvs = np.array(sub_obvs)
        sub_n_obvs = np.array(sub_n_obvs)
        sub_acts = np.array(sub_acts)
        sub_rews = np.array(sub_rews)
        sub_dones = list(sub_dones)

        if c.learn_model_primitive and c.model_primitive_pg:
            self.sess.run(self.update_old_mp_pi_op)

            mp_acts = np.array([i['mp_act'] for i in sub_infos])
            mp_baseline = np.zeros(mp_acts.shape[0])
            mp_rews = -vector_l2_loss_np(mp_acts + sub_obvs if c.model_primitive_delta else mp_acts, sub_n_obvs)
            self.mp_rewards = np.mean(mp_rews)
            mp_dones = np.ones(mp_acts.shape[0]).tolist()

            mp_seg = generate_seg(mp_dones, mp_rews, mp_baseline)
            add_vtarg_and_adv(mp_seg)
            mp_advs, mp_vtarg = mp_seg["adv"], mp_seg["tdlamret"]
            mp_advs = (mp_advs - mp_advs.mean()) / (mp_advs.std())

        # NOTE: BASELINE PRE, UPDATE OLD POLICY
        sub_baseline, run_std, _ = self.sess.run([self.bl_tr, self.ob_rms.std, self.assign_old_sub_op], feed_dict={
            self.obv_ph_raw: sub_obvs
        })

        # NOTE: ADVANGTAGES
        sub_seg = generate_seg(sub_dones, sub_rews, sub_baseline)
        add_vtarg_and_adv(sub_seg)
        sub_advs, sub_vtarg = sub_seg["adv"], sub_seg["tdlamret"]
        sub_advs = (sub_advs - sub_advs.mean()) / (sub_advs.std())
        self.improved_variance = explained_variance(sub_baseline, sub_vtarg)

        if c.domain == 'stacker' and c.split:
            self.ob_rms.update(sub_obvs[:, :-self.env.stage_vector_size], self.sess)
        else:
            self.ob_rms.update(sub_obvs, self.sess)
        num_time_steps = sub_obvs.shape[0]

        # NOTE: SHUFFLING
        bs = c.mini_bs
        num_iter = int(num_time_steps // bs)
        order = np.random.permutation(num_time_steps).tolist()

        sub_obvs = sub_obvs[order]
        sub_acts = sub_acts[order]
        sub_n_obvs = sub_n_obvs[order]
        sub_advs = sub_advs[order]
        sub_vtarg = sub_vtarg[order]

        if c.learn_model_primitive and c.model_primitive_pg:
            mp_advs = mp_advs[order]
            mp_acts = mp_acts[order]

        # NOTE: NEXT S PROBABILITIES
        self.pred_gt_obvs = self.calc_post_belief(sub_obvs, sub_acts, sub_n_obvs, sub_infos)
        self.fd = {
            self.obv_ph_raw: sub_obvs,
            self.sub_act_ph: sub_acts,
            self.sub_adv_ph: sub_advs,
            self.n_obv_ph: sub_n_obvs,
            self.bl_target_ph: sub_vtarg,
            self.pred_gt_obvs_ph: self.pred_gt_obvs,
            self.oracle_m_label_ph: self.true_model_index,
            self.sub_lr_ph: self.sub_lr,
            self.lr_mult_ph: self.lr_mult,
            self.master_pg_lr_ph: self.master_pg_lr,
            self.master_ce_lr_ph: self.master_ce_lr,
            self.oracle_master_ph: self.oracle_or_not(),
            self.mp_lr_ph: self.mp_lr,
            self.running_std_text_ph: ', '.join([str(ii) for ii in run_std]),
            self.cor_width_ph: str(c.cor_wd),
            self.mp_ckpt_ph: str(c.model_primitive_checkpoints)
        }

        # NOTE: Train Subpolicy
        if c.num_models > 1:
            self.master_ce_loss, master_ce_loss_raw = self.sess.run(
                [self.master_ce_loss_tr, self.master_ce_loss_raw_tr],
                feed_dict=self.fd)

            safe = (master_ce_loss_raw < c.safe_m_loss) if \
                self.transfer_learning() else np.ones(num_time_steps)
        else:
            safe = np.ones(num_time_steps)

        indices = np.array([i for i in range(num_time_steps) if safe[i]])
        self.good_sub_num_samples = np.sum(safe)
        num_iter_sub = int(self.good_sub_num_samples // bs)

        self.fd[self.mean_safe_perc_tr] = np.mean(safe)

        sub_batches = []
        for i in range(num_iter_sub):
            order = range(i * bs, (i + 1) * bs)
            real_ord = indices[order].tolist()

            sub_batch_fd = {
                self.obv_ph_raw: sub_obvs[real_ord],
                self.sub_act_ph: sub_acts[real_ord],
                self.sub_adv_ph: sub_advs[real_ord],
                self.bl_target_ph: sub_vtarg[real_ord],
                self.pred_gt_obvs_ph: self.pred_gt_obvs[real_ord],
                self.oracle_m_label_ph: self.true_model_index[real_ord],
                self.sub_lr_ph: self.sub_lr,
                self.lr_mult_ph: self.lr_mult,
                self.bl_lr_ph: self.bl_lr,
                self.master_pg_lr_ph: self.master_pg_lr,
                self.master_ce_lr_ph: self.master_ce_lr,
                self.mp_lr_ph: self.mp_lr,
                self.oracle_master_ph: self.oracle_or_not()
            }
            if c.learn_model_primitive:
                if c.model_primitive_pg:
                    sub_batch_fd[self.mp_adv_ph] = mp_advs[real_ord]
                    sub_batch_fd[self.mp_act_ph] = mp_acts[real_ord]
                else:
                    sub_batch_fd[self.n_obv_ph] = sub_n_obvs[real_ord]

            sub_batches.append(sub_batch_fd)

        self.subpolicy_loss = []

        train_ops = [self.sub_surr_loss_tr, self.train_sub_op[self.task_id]]
        if self.fd[self.mean_safe_perc_tr] == 1:
            train_ops.append(self.train_bl_op[self.task_id])
        if c.learn_model_primitive:
            if c.model_primitive_pg:
                self.fd[self.mp_adv_ph] = mp_advs
                self.fd[self.mp_act_ph] = mp_acts
            train_ops += [self.mp_loss_tr, self.train_mp_op]

        for i in range(c.optim_epochs):
            for batch in sub_batches:
                sub_loss = self.sess.run(train_ops, feed_dict=batch)[0]
                if i == c.optim_epochs - 1 and sub_baseline != []:
                    self.subpolicy_loss.append(sub_loss)
        self.subpolicy_loss = np.mean(self.subpolicy_loss) if self.subpolicy_loss else 0
        # NOTE: END

        self.fd[self.timesteps_used_text_ph] = str(self.timesteps_used_list)
        self.fd[self.ts_string_ph] = ' & '.join([str(round(num, 1)) for num in self.timesteps_used_list])

        self.policy_kl_divergence, _, = self.sess.run([self.policy_kl_divergence_tr, self.assign_old_sub_op],
                                                      feed_dict=self.fd)

        if socket.gethostname() != 'tula' and self.is_mujoco():
            self.im = np.flip(self.env.env.sim.render(c.gif_s, c.gif_s), axis=0)
            self.fd[self.image] = [self.im]
        if c.num_models == 1:
            return

        # NOTE: TRAIN MASTER AND BASELINE NETWORK
        m_batches = []
        for i in range(num_iter):
            order = range(i * bs, (i + 1) * bs)
            sub_batch_fd = {
                self.obv_ph_raw: sub_obvs[order],
                self.sub_act_ph: sub_acts[order],
                self.sub_adv_ph: sub_advs[order],
                self.bl_target_ph: sub_vtarg[order],
                self.pred_gt_obvs_ph: self.pred_gt_obvs[order],
                self.oracle_m_label_ph: self.true_model_index[order],
                self.sub_lr_ph: self.sub_lr,
                self.lr_mult_ph: self.lr_mult,
                self.bl_lr_ph: self.bl_lr,
                self.master_pg_lr_ph: self.master_pg_lr,
                self.master_ce_lr_ph: self.master_ce_lr,
                self.oracle_master_ph: self.oracle_or_not()
            }
            m_batches.append(sub_batch_fd)

        if (self.transfer_learning() and c.enforced) or c.restore_model:
            train_ops = [self.train_bl_op[self.task_id]] if self.fd[self.mean_safe_perc_tr] < 1 else []
            if c.num_models > 1:
                train_ops.append(self.train_master_ce_op[self.task_id])
                if not c.old_policy and c.pg_master_pg_lr > 0:
                    train_ops.append(self.train_master_pg_op[self.task_id])

            for i in range(c.optim_epochs):
                for batch in m_batches:
                    self.sess.run(train_ops, feed_dict=batch)
        else:
            train_ops = [self.train_master_ce_op[self.task_id]] if c.num_models > 1 else []
            if not c.old_policy and c.num_models > 1 and c.pg_master_pg_lr > 0:
                train_ops.append(self.train_master_pg_op[self.task_id])

            if self.fd[self.mean_safe_perc_tr] < 1:
                for i in range(c.optim_epochs):
                    for batch in m_batches:
                        self.sess.run(self.train_bl_op[self.task_id], feed_dict=batch)
            self.sess.run(train_ops, feed_dict=self.fd)
        # NOTE: END

        # NOTE: UPDATE MASTER
        if (self.t + 1) % c.update_master_interval == 0 and c.num_models > 1:
            print('Updating Master and Math Subpolicy Network...')
            if c.math:
                self.sess.run([self.update_target_master_op, self.update_math_sub_op])
            else:
                self.sess.run(self.update_target_master_op)
            print('Updated')
        # NOTE: END

    def add_update_target_master_op(self):
        self.curr_master_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curr_master')
        self.target_master_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_master')
        assign_ops = []
        for i in range(len(self.curr_master_vars)):
            assign_ops.append(
                tf.assign(self.target_master_vars[i], self.curr_master_vars[i]))
        self.update_target_master_op = tf.group(assign_ops)

    def calc_kl_divergence(self, d1, d2):
        return tf.reduce_sum(d2['act_log_std'] - d1['act_log_std'] + (
                tf.square(d1['action_std']) + tf.square(
            d1['mean'] - d2['mean'])) / (2.0 * tf.square(
            d2['action_std'])) - 0.5, axis=-1)

    def add_update_math_subpolicy_op(self):
        self.curr_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curr_pi')
        self.math_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='math_pi')
        assign_ops = []
        for i in range(len(self.curr_pi_vars)):
            curr_pi_var = self.curr_pi_vars[i]
            math_pi_var = self.math_pi_vars[i]
            assign_ops.append(tf.assign(math_pi_var, curr_pi_var))
        self.update_math_sub_op = tf.group(assign_ops)

    def add_update_old_subpolicy_op(self):
        self.policy_kl_divergence_tr = tf.reduce_mean(
            self.calc_kl_divergence(self.curr_pi, self.old_pi))
        self.curr_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curr_pi')
        self.old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pi')
        assign_ops = []
        for i in range(len(self.curr_pi_vars)):
            curr_pi_var = self.curr_pi_vars[i]
            old_pi_var = self.old_pi_vars[i]
            assign_ops.append(tf.assign(old_pi_var, curr_pi_var))
        self.assign_old_sub_op = tf.group(assign_ops)

    def calc_post_belief(self, obvs, acts, next_obvs, info):
        obvs = np.array(obvs)
        acts = np.array(acts)
        next_obvs = np.array(next_obvs)

        true_next_obvs = np.repeat(next_obvs[:, np.newaxis, :], repeats=c.num_models, axis=1)

        raw = self.calc_true_model_index(info)
        if c.xml_name.startswith(('C', '5')):
            corr_model_mask = raw
            self.true_model_index = np.argmax(raw, axis=1)
        else:
            if c.num_models > 1:
                self.true_model_index = raw.astype(np.int64)
            else:
                self.true_model_index = np.zeros_like(raw)

            corr_model_mask = np.zeros((obvs.shape[0], c.num_models))
            corr_model_mask[np.arange(obvs.shape[0]), self.true_model_index] = 1

        if c.use_model_primitive:
            noisy_next_obvs_list = self.sess.run(self.next_s_pred_tr_list, feed_dict={
                self.obv_ph_raw: obvs, self.sub_act_ph: acts
            })

            noisy_next_obvs = np.stack(noisy_next_obvs_list, axis=1)
        else:
            corr_model_mask = corr_model_mask[:, :, np.newaxis]
            true_model_noise = c.Tnoise * corr_model_mask * c.run_std * np.random.randn(*true_next_obvs.shape)
            false_model_noise = c.Fnoise * (1 - corr_model_mask) * c.run_std * np.random.randn(*true_next_obvs.shape)
            noisy_next_obvs = true_next_obvs + true_model_noise + false_model_noise

        return np.concatenate([noisy_next_obvs[:, :, np.newaxis, :], true_next_obvs[:, :, np.newaxis, :]], axis=2)

    def add_placeholders_op(self):
        self.obv_ph_raw = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="obv_ph_raw")
        self.n_obv_ph = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="n_obv_ph")

        with tf.variable_scope("obfilter"):
            if c.domain == 'stacker' and c.split:
                unnorm_size = self.env.stage_vector_size
                self.ob_rms = RunningMeanStd(shape=self.obs_dim - unnorm_size)
                self.obv_ph = tf.concat([tf.clip_by_value(
                    (self.obv_ph_raw[:, :-unnorm_size] - self.ob_rms.mean) / self.ob_rms.std, -5, 5),
                    self.obv_ph_raw[:, -unnorm_size:]], axis=1)
            else:
                self.ob_rms = RunningMeanStd(shape=self.obs_dim)
                self.obv_ph = tf.clip_by_value((self.obv_ph_raw - self.ob_rms.mean) / self.ob_rms.std, -5, 5)
        self.init_obfilter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope='obfilter')
        self.reset_obfilter_op = tf.variables_initializer(
            var_list=self.init_obfilter)
        if self.discrete:
            self.sub_act_ph = tf.placeholder(tf.int64, shape=None, name='sub_act_ph')
        else:
            self.sub_act_ph = tf.placeholder(tf.float32, shape=[None, self.act_dim], name='sub_act_ph')
        self.sub_adv_ph = tf.placeholder(tf.float32, shape=None, name='sub_adv_ph')
        self.bl_target_ph = tf.placeholder(tf.float32, shape=None, name='bl_target_ph')
        self.bl_lr_ph = tf.placeholder(tf.float32, shape=None, name='bl_lr_ph')
        self.master_ce_lr_ph = tf.placeholder(tf.float32, shape=None, name='master_ce_lr_ph')
        self.sub_lr_ph = tf.placeholder(tf.float32, shape=None, name='sub_lr_ph')
        self.mp_lr_ph = tf.placeholder(tf.float32, shape=None, name='mp_lr_ph')
        self.lr_mult_ph = tf.placeholder(tf.float32, shape=None, name='lr_mult_ph')

        self.master_pg_lr_ph = tf.placeholder(tf.float32, shape=None, name='master_pg_lr_ph')

        self.pred_gt_obvs_ph = tf.placeholder(tf.float32, shape=[None, c.num_models, 2,
                                                                 self.obs_dim], name='pred_gt_obvs_ph')
        self.mean_safe_perc_tr = tf.placeholder(tf.float32, name='mean_safe_perc_tr')
        self.oracle_m_label_ph = tf.placeholder(tf.int64, shape=[None])
        self.oracle_master_ph = tf.placeholder(tf.float32, shape=[1])
        self.oracle_master_vis_tr = self.oracle_master_ph[0]

    def add_baseline_op(self, scope="baseline"):
        self.bl_tr = tf.squeeze(
            dnn(input=self.obv_ph, output_size=1,
                scope=scope, n_layers=c.pg_bl_n_layers,
                size=c.pg_bl_hidden_fc_size, hid_init=normc_initializer(1.0),
                final_init=normc_initializer(1.0)))

    def custom_init_op(self):
        assert not (c.learn_model_primitive and c.use_model_primitive)

        self.timesteps_used_list = []
        self.timesteps_used_text_ph = str([])
        self.clip_param_tr = c.clip_param
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.obs_dim = 1 if self.discrete else self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        self.add_placeholders_op()
        self.add_build_master_op()
        if c.learn_model_primitive or c.use_model_primitive:
            self.add_model_primitve_op()
        self.curr_pi = self.build_policy_network_op(scope='curr_pi', trainable=self.main and not c.fix_sub)
        self.add_build_master_for_train_op()
        if self.main:
            self.add_baseline_op()
            if c.num_models > 1:
                self.master_entropy_tr = tf.reduce_mean(self.curr_pi['entropy'])
                self.tensorboard_keys += ['vars/master_entropy_tr']
            self.old_pi = self.build_policy_network_op(scope='old_pi', trainable=False)
            if c.math:
                self.math_pi = self.build_policy_network_op(scope='math_pi', trainable=False)
                self.add_update_math_subpolicy_op()
            self.add_update_old_subpolicy_op()
            self.add_update_target_master_op()
            self.add_train_op()
            self.tensorboard_keys += ['loss/bl_loss_tr',
                                      'loss/subpolicy_loss',
                                      'loss/ent_loss_tr',
                                      'percentage/mean_safe_perc_tr',
                                      'vars/improved_variance',
                                      'vars/good_sub_num_samples',
                                      'vars/policy_kl_divergence',
                                      'vars/oracle_master_vis_tr',
                                      'rewards/task_id',
                                      'vars/dx_mean',
                                      'vars/dx_std',
                                      'vars/dy_mean',
                                      'vars/dy_std',
                                      'rewards/survival_reward',
                                      'rewards/forward_reward',
                                      'rewards/control_reward',
                                      'rewards/contact_reward',
                                      'rewards/distance_to_goal',
                                      'lr/master_ce_lr',
                                      'lr/master_pg_lr',
                                      'lr/sub_lr',
                                      'lr/bl_lr',
                                      'timesteps/iteration_solved',
                                      'grad_norm/pi_grad_norm_tr',
                                      'grad_norm/bl_grad_norm_tr',
                                      'grad_norm/clip_param_tr',
                                      'sys/timesteps_so_far']

            for i in range(c.num_models):
                self.tensorboard_keys += ['percentage/suc_test_sub_%s' % i,
                                          'timesteps/ts_test_sub_%s' % i,
                                          'rewards/reward_test_sub_%s' % i,
                                          'percentage/sub_%s_perc_tr' % i]
            for key in self.curr_pi:
                if key.startswith('kl_divergence'):
                    setattr(self, key, self.curr_pi[key])
                    self.tensorboard_keys.append('vars/%s' % key)

            self.bl_lr = c.pg_bl_lr
            self.sub_lr = c.pg_sub_lr
            self.master_ce_lr = c.pg_master_ce_lr_source

            self.image = tf.placeholder(dtype=tf.uint8, shape=(1, c.gif_s, c.gif_s, 3))
            self.running_std_text_ph = tf.placeholder(dtype=tf.string)
            self.timesteps_used_text_ph = tf.placeholder(dtype=tf.string)
            self.mp_ckpt_ph = tf.placeholder(dtype=tf.string)
            self.ts_string_ph = tf.placeholder(dtype=tf.string)
            self.cor_width_ph = tf.placeholder(dtype=tf.string)

            tf.summary.text('run_std', self.running_std_text_ph)
            tf.summary.text('timesteps_used_text', self.timesteps_used_text_ph)
            tf.summary.text('ts_string', self.ts_string_ph)
            tf.summary.text('mp_ckpt', self.mp_ckpt_ph)
            tf.summary.text('corridor_width', self.cor_width_ph)

            if socket.gethostname() != 'tula' and self.is_mujoco():
                tf.summary.image('image', self.image)

        non_sub_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='^((?!policy).)*$')
        self.reset_non_sub_op = tf.variables_initializer(var_list=non_sub_vars)

    def evaluate_avg_r(self):
        self.save_model_op()
        ray.get([actor.restore_model_op.remote(self.run_id) for actor in self.actors])
        id = np.random.randint(0, c.num_cores)

        ret = zip(*ray.get(
            [self.actors[i].evaluate_one_rollout.remote(id == i) for i in
             range(c.num_cores)]))

        r, frames, num_succ_mas, t, forw_r, surv_r, ctrl_r, cont_r, info = ret

        num_succ_mas = np.array(num_succ_mas)
        self.eval_timesteps = (np.array(t) * num_succ_mas).sum() / (num_succ_mas.sum() + c.e)
        self.suc_test = num_succ_mas.mean()
        self.suc_test_std = num_succ_mas.std()
        self.survival_reward = np.mean(surv_r)
        self.forward_reward = np.mean(forw_r)
        self.control_reward = np.mean(ctrl_r)
        self.contact_reward = np.mean(cont_r)
        self.distance_to_goal = np.mean([i['dist'] for i in info])

        if c.num_models > 1:
            bs = c.num_cores // c.num_models
            sub_r, _, num_succ, t, _, _, _, _, _ = zip(*ray.get(
                [self.actors[i].evaluate_one_rollout.remote(id == i,
                                                            sub=i // bs)
                 for i in range(bs * c.num_models)]))

            num_succ = np.array(num_succ)
            sub_r = np.array(sub_r)
            t = np.array(t)

            for j in range(c.num_models):
                order = range(j * bs, (j + 1) * bs)
                setattr(self, 'suc_test_sub_%s' % j, num_succ[order].mean())
                setattr(self, 'reward_test_sub_%s' % j, sub_r[order].mean())
                setattr(self, 'ts_test_sub_%s' % j,
                        (t[order] * num_succ[order]).sum() / (
                                num_succ[order].sum() + c.e))
        else:
            self.suc_test_sub_0, self.reward_test_sub_0, self.ts_test_sub_0 = \
                0, 0, 0
        return np.mean(r)

    def build_policy_network_op(self, scope="policy", trainable=True):
        ac_means = []
        ac_log_stds = []
        ac_stds = []
        dists = []
        pi = {}
        with tf.variable_scope(scope):
            for i in range(c.num_models):
                with tf.variable_scope('policy_%s' % i):
                    if self.discrete:
                        raise NotImplementedError
                    else:
                        ac_means.append(dnn(input=self.obv_ph,
                                            output_size=self.act_dim,
                                            scope='dnn',
                                            n_layers=c.pg_pi_n_layers,
                                            size=c.pg_pi_hidden_fc_size,
                                            trainable=trainable,
                                            hid_init=normc_initializer(1.0),
                                            final_init=normc_initializer(1.0)))
                        ac_log_stds.append(tf.get_variable('act_log_std_%s' % i,
                                                           shape=[self.act_dim],
                                                           trainable=trainable,
                                                           initializer=tf.zeros_initializer()))
                        ac_stds.append(tf.exp(ac_log_stds[i]))
                        dists.append(tfd.MultivariateNormalDiag(loc=ac_means[i],
                                                                scale_diag=tf.zeros_like(ac_means[i]) + ac_stds[i]))
                        pi['sampled_action_sub_%s' % i] = tf.squeeze(dists[i].sample())
                        pi['logprob_s_%s' % i] = dists[i].log_prob(self.sub_act_ph)

            for i in range(c.num_models):
                for j in range(i + 1, c.num_models):
                    pi['kl_divergence_%s_%s_tr' % (
                        i, j)] = tf.reduce_mean(
                        tfd.kl_divergence(distribution_a=dists[i], distribution_b=dists[j], allow_nan_stats=False))

            oracle_master_tr = tf.tile(
                tf.expand_dims(self.oracle_master_ph, axis=1),
                (1, c.num_models))  # NOTE: [1] or [0]

            if c.num_models == 1:
                pi['sampled_action'] = dists[0].sample()
                pi['logprob'] = dists[0].log_prob(self.sub_act_ph)
                pi['action_std'] = dists[0].stddev()
                pi['mean'] = dists[0].mean()
                pi['act_log_std'] = tf.log(pi['action_std'])
                return pi

            if c.oracle_master:
                probs = tf.one_hot(self.oracle_m_label_ph,  # NOTE: ORACLE
                                   depth=c.num_models) * oracle_master_tr + (
                                1 - oracle_master_tr) * self.master_prob  #
                # NOTE: SOFT
            else:
                probs = self.master_prob

            categorial = tfd.Categorical(probs=probs)
            pi['entropy'] = categorial.entropy()
            gaussian_mixture = tfd.Mixture(cat=categorial, components=dists)

            pi['sampled_action'] = tf.squeeze(gaussian_mixture.sample())
            pi['logprob'] = gaussian_mixture.log_prob(self.sub_act_ph)
            pi['action_std'] = gaussian_mixture.stddev()
            pi['act_log_std'] = tf.log(pi['action_std'])
            pi['mean'] = gaussian_mixture.mean()

        return pi

    def add_master_ce_train_op(self):
        # NOTE: LOSS
        self.master_ce_loss_raw_tr = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(self.master_posterior),
            logits=self.master_logits_curr)
        if c.weighted:
            self.master_ce_loss_tr = tf.reduce_mean(tf.reduce_sum(
                self.master_ce_loss_raw_tr[:, tf.newaxis] * self.label_one_hot,
                axis=0) / self.label_count)
        else:
            self.master_ce_loss_tr = tf.reduce_mean(self.master_ce_loss_raw_tr)

        # NOTE: OPTIMIZER
        self.train_master_ce_op = []
        for i in range(c.num_tasks):
            master_ce_opt = tf.train.AdamOptimizer(
                learning_rate=self.master_ce_lr_ph, epsilon=1e-5)
            self.train_master_ce_op.append(slim.learning.create_train_op(
                self.master_ce_loss_tr, master_ce_opt,
                clip_gradient_norm=c.gradclip,
                summarize_gradients=False,
                check_numerics=True))

        master_ce_opt = tf.train.AdamOptimizer(learning_rate=self.master_ce_lr_ph, epsilon=1e-5)
        master_ce_grad_and_vars = master_ce_opt.compute_gradients(self.master_ce_loss_tr)
        master_ce_grad = [i[0] for i in master_ce_grad_and_vars if
                          i[0] is not None]
        self.master_ce_grad_norm_tr = tf.global_norm(master_ce_grad)

        self.tensorboard_keys += ['grad_norm/master_ce_grad_norm_tr']

    def add_master_pg_train_op(self):
        # NOTE: OPTIMIZER
        self.train_master_pg_op = []
        for i in range(c.num_tasks):
            master_pg_opt = tf.train.AdamOptimizer(learning_rate=self.master_pg_lr_ph, epsilon=1e-5)
            self.train_master_pg_op.append(slim.learning.create_train_op(
                self.sub_loss_tr, master_pg_opt,
                variables_to_train=self.curr_master_vars,
                clip_gradient_norm=c.gradclip,
                check_numerics=True,
                summarize_gradients=False))
        master_pg_opt = tf.train.AdamOptimizer(
            learning_rate=self.master_pg_lr_ph, epsilon=1e-5)
        master_pg_grad_and_vars = master_pg_opt.compute_gradients(self.sub_loss_tr)
        master_pg_grad = [i[0] for i in master_pg_grad_and_vars if i[0] is not None]
        self.master_pg_grad_norm_tr = tf.global_norm(master_pg_grad)
        self.tensorboard_keys += ['grad_norm/master_pg_grad_norm_tr']

    def add_subpolicy_train_op(self):
        # NOTE: LOSS
        self.ent_loss_tr = -tf.reduce_sum(
            self.curr_pi['act_log_std'] + .5 * np.log(2.0 * np.pi * np.e))
        self.ratio_tr = tf.exp(
            self.curr_pi['logprob'] - tf.stop_gradient(self.old_pi['logprob']))
        surr1 = self.ratio_tr * self.sub_adv_ph
        self.clip_param_tr = c.clip_param * self.lr_mult_ph
        surr2 = tf.clip_by_value(self.ratio_tr, 1.0 - self.clip_param_tr, 1.0 + self.clip_param_tr) * self.sub_adv_ph
        self.sub_surr_loss_tr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        self.sub_loss_tr = self.sub_surr_loss_tr + self.ent_loss_tr * c.ent_coeff
        # NOTE: OPTIMIZER
        self.train_sub_op = []
        for i in range(c.num_tasks):
            pi_opt = tf.train.AdamOptimizer(learning_rate=self.sub_lr_ph, epsilon=1e-5)
            self.train_sub_op.append(
                slim.learning.create_train_op(self.sub_loss_tr,
                                              pi_opt,
                                              variables_to_train=self.curr_pi_vars,
                                              clip_gradient_norm=c.gradclip,
                                              check_numerics=True,
                                              summarize_gradients=False))

        pi_opt = tf.train.AdamOptimizer(learning_rate=self.sub_lr_ph, epsilon=1e-5)
        pi_grad_and_vars = pi_opt.compute_gradients(self.sub_loss_tr)
        self.pi_grad_norm_tr = tf.global_norm([i[0] for i in pi_grad_and_vars if i[0] is not None])

    def add_baseline_train_op(self):
        # NOTE: LOSS
        self.bl_loss_tr = tf.reduce_mean(
            tf.square(self.bl_target_ph - self.bl_tr))

        # NOTE: OPTIMIZER
        self.train_bl_op = []
        for i in range(c.num_tasks):
            bl_opt = tf.train.AdamOptimizer(learning_rate=self.bl_lr_ph,
                                            epsilon=1e-5)
            self.train_bl_op.append(
                slim.learning.create_train_op(self.bl_loss_tr,
                                              bl_opt,
                                              check_numerics=True,
                                              clip_gradient_norm=c.gradclip,
                                              summarize_gradients=False))

        bl_opt = tf.train.AdamOptimizer(learning_rate=self.bl_lr_ph, epsilon=1e-5)
        bl_grad_and_vars = bl_opt.compute_gradients(self.bl_loss_tr)
        self.bl_grad_norm_tr = tf.global_norm([i[0] for i in bl_grad_and_vars if i[0] is not None])

    def add_train_op(self):
        if c.num_models > 1:
            self.add_master_ce_train_op()
            self.tensorboard_keys += ['loss/master_ce_loss_tr', 'loss/master_ce_loss']
        self.add_subpolicy_train_op()
        self.add_baseline_train_op()
        if not c.old_policy and c.num_models > 1 and c.pg_master_pg_lr > 0:
            self.add_master_pg_train_op()

    def add_build_master_op(self, scope='master'):
        # NOTE: MASTER NETWORK
        with tf.variable_scope('target_master'):
            self.master_logits_old = dnn(input=self.obv_ph,
                                         output_size=c.num_models,
                                         scope=scope,
                                         n_layers=c.pg_master_n_layers,
                                         size=c.pg_master_hidden_fc_size,
                                         trainable=False)
            self.master_prob_old = tf.nn.softmax(logits=self.master_logits_old, axis=1)

        with tf.variable_scope('curr_master'):
            self.master_logits_curr = dnn(input=self.obv_ph,
                                          output_size=c.num_models,
                                          scope=scope,
                                          n_layers=c.pg_master_n_layers,
                                          size=c.pg_master_hidden_fc_size,
                                          trainable=self.main)
            self.master_prob_curr = tf.nn.softmax(logits=self.master_logits_curr, axis=1)

            self.master_prob_selected = self.master_prob_old if c.old_policy else self.master_prob_curr

            if c.soft:
                self.master_prob = self.master_prob_selected
            else:
                self.master_prob = tf.one_hot(indices=tf.argmax(self.master_prob_selected, axis=1), depth=c.num_models)

            self.master_prob_avg = tf.reduce_mean(self.master_prob_selected,
                                                  axis=0)
            for i in range(c.num_models):
                setattr(self, 'sub_%s_perc_tr' % i, self.master_prob_avg[i])

    def add_build_master_for_train_op(self):
        # NOTE: MASTER
        next_s_log_probs = []
        for i in range(c.num_models):
            multivar_gau_dist = tfd.MultivariateNormalDiag(
                loc=self.pred_gt_obvs_ph[:, i, 0],
                scale_diag=tf.zeros_like(self.pred_gt_obvs_ph[:, i, 0]) + c.run_std,
                validate_args=True,
                allow_nan_stats=False)
            next_s_log_probs.append(multivar_gau_dist.log_prob(self.pred_gt_obvs_ph[:, i, 1]))

        self.ns_logprobs_tr = tf.stack(next_s_log_probs, axis=1)

        if c.math:
            ac_logprob_list = []
            for i in range(c.num_models):
                if c.stable_old:
                    ac_logprob_list.append(self.old_pi['logprob_s_%s' % i])
                else:
                    ac_logprob_list.append(self.curr_pi['logprob_s_%s' % i])
            self.ac_logprob = tf.stop_gradient(tf.stack(ac_logprob_list, axis=1))

            self.m_log_posterior = self.ns_logprobs_tr + self.ac_logprob + tf.log(self.master_prob_old)
        else:
            self.m_log_posterior = self.ns_logprobs_tr + tf.log(self.master_prob_old)

        self.m_log_posterior -= tf.reduce_max(self.m_log_posterior, axis=1,
                                              keepdims=True)
        self.master_posterior_raw = tf.exp(self.m_log_posterior) + 1e-10
        if c.l2:
            self.master_posterior = tf.nn.l2_normalize(self.master_posterior_raw, axis=1)
        else:
            self.master_posterior_raw_sum = tf.reduce_sum(self.master_posterior_raw, axis=1, keepdims=True)
            self.master_posterior = self.master_posterior_raw / self.master_posterior_raw_sum
        self.label_one_hot = tf.one_hot(
            indices=tf.argmax(self.master_posterior, axis=1), axis=1,
            depth=c.num_models)
        self.label_count = tf.reduce_sum(self.label_one_hot, axis=0) + c.e

    def update_learning_rates_op(self, t):
        self.t = t
        self.timesteps_so_far += self.num_timesteps
        self.eps = float(self.egreedy_schedule.epsilon)
        self.alpha = float(self.alpha_schedule.epsilon)
        self.lr_mult = max(1.0 - self.timesteps_so_far / c.total_ts, 0.01)
        self.ce_lr_mult = max(1.0 - self.timesteps_so_far / c.total_ts, 0.0)
        self.master_pg_lr = c.pg_master_pg_lr * self.ce_lr_mult * (
                self.timesteps_so_far >= c.stop_pg_ts)
        self.master_ce_lr = (c.pg_master_ce_lr_target if
                             self.transfer_learning() else
                             c.pg_master_ce_lr_source) * self.ce_lr_mult * (
                                    self.timesteps_so_far < c.stop_pg_ts)
        self.bl_lr = c.pg_bl_lr * self.lr_mult
        self.sub_lr = c.pg_sub_lr * self.lr_mult
        self.mp_lr = c.pg_mp_lr * self.lr_mult

    def build_mp_network_op(self, scope, input, output_dim, act_ph, n_layers, hid_fc_size, trainable):
        pi = {}
        with tf.variable_scope(scope):
            pi['raw_mean'] = dnn_mp(input, output_dim, 'dnn', n_layers, hid_fc_size, trainable,
                                    normc_initializer(1.0), normc_initializer(1.0))

            pi['mean'] = pi['raw_mean'] if c.model_primitive_delta else pi['delta_mean'] + self.obv_ph_raw
            pi['log_std'] = tf.get_variable('act_log_std', shape=[output_dim], trainable=trainable,
                                            initializer=normc_initializer(0.01))
            pi['std'] = tf.exp(pi['log_std'])

            dist = tfd.MultivariateNormalDiag(loc=pi['mean'], scale_diag=tf.zeros_like(pi['mean']) + pi['log_std'])
            pi['sampled_action'] = tf.squeeze(dist.sample())

            pi['log_prob'] = dist.log_prob(act_ph)

            if not trainable:
                with tf.variable_scope('freeze_old_pi'):
                    for key in pi:
                        pi[key] = tf.stop_gradient(pi[key])

        return pi

    def update_old_pi_op(self, i):
        curr_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='curr_model_primitive_%s' % i)
        old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_model_primitive_%s' % i)
        assert len(curr_pi_vars) == len(old_pi_vars), (curr_pi_vars, old_pi_vars)
        for i in range(len(curr_pi_vars)):
            assert old_pi_vars[i].name[6:] == curr_pi_vars[i].name[7:]
        assign_ops = [tf.assign(old_pi_vars[i], curr_pi_vars[i]) for i in range(len(curr_pi_vars))]
        update_old_pi_op = tf.group(*assign_ops)
        return curr_pi_vars, old_pi_vars, update_old_pi_op

    def train_op(self, loss_tr, lr_ph, scope):
        with tf.variable_scope(scope):
            opt = tf.train.AdamOptimizer(learning_rate=lr_ph, epsilon=1e-5)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            train_op = slim.learning.create_train_op(loss_tr, opt, variables_to_train=vars,
                                                     clip_gradient_norm=c.gradclip, check_numerics=True,
                                                     summarize_gradients=True)
            grad_and_vars = opt.compute_gradients(loss_tr, var_list=vars)
            grad_norm_tr = tf.global_norm([i[0] for i in grad_and_vars if i[0] is not None])
        return train_op, grad_norm_tr

    def build_pi_train_op(self, i, curr_pi, old_pi, adv_ph, lr_ph):
        with tf.variable_scope('%s_surrogate_loss' % i):
            log_ratio_tr = curr_pi['log_prob'] - old_pi['log_prob']
            ratio_tr = tf.exp(log_ratio_tr)
            clip_param_tr = c.clip_param * self.lr_mult_ph
            surr1 = ratio_tr * adv_ph
            surr2 = tf.clip_by_value(ratio_tr, 1.0 - clip_param_tr, 1.0 + clip_param_tr) * adv_ph
            pi_loss_tr = -tf.reduce_mean(tf.minimum(surr1, surr2))

        train_pi_op, pi_grad_norm_tr = self.train_op(pi_loss_tr, lr_ph, 'curr_model_primitive_%s' % i)
        return train_pi_op, pi_grad_norm_tr, pi_loss_tr, {'ratio_tr': ratio_tr}

    def add_model_primitve_op(self):
        self.s_a_tr = tf.concat(values=[self.obv_ph_raw, self.sub_act_ph], axis=1, name='s_a_tr')
        if c.use_model_primitive:
            self.model_primitives = []
            self.next_s_pred_tr_list = []
            for i in range(c.num_models):
                with tf.variable_scope('curr_model_primitive_%s' % i):
                    pred = dnn(input=self.s_a_tr, output_size=self.obs_dim, scope='next_s_pred_tr',
                               n_layers=c.pg_mp_n_layers, size=c.pg_mp_hidden_fc_size, trainable=False)
                    self.next_s_pred_tr_list.append(pred)
        else:
            if c.model_primitive_pg:
                self.tensorboard_keys += ['rewards/mp_rewards', 'loss/mp_loss_tr']
                self.mp_act_ph = tf.placeholder(tf.float32, (None, self.obs_dim), name='mp_act_ph')
                self.mp_adv_ph = tf.placeholder(tf.float32, name='mp_adv_ph')
                self.curr_mp_pi = self.build_mp_network_op(scope='curr_model_primitive_%s' % c.xml_name[1],
                                                           input=self.s_a_tr, output_dim=self.obs_dim,
                                                           act_ph=self.mp_act_ph, n_layers=c.pg_mp_n_layers,
                                                           hid_fc_size=c.pg_mp_hidden_fc_size, trainable=True)
                self.old_mp_pi = self.build_mp_network_op(scope='old_model_primitive_%s' % c.xml_name[1],
                                                          input=self.s_a_tr, output_dim=self.obs_dim,
                                                          act_ph=self.mp_act_ph, n_layers=c.pg_mp_n_layers,
                                                          hid_fc_size=c.pg_mp_hidden_fc_size, trainable=False)
                self.curr_mp_pi_vars, self.old_mp_pi_vars, self.update_old_mp_pi_op = self.update_old_pi_op(
                    c.xml_name[1])
                self.train_mp_op, self.mp_grad_norm_tr, self.mp_loss_tr, _ = self.build_pi_train_op(c.xml_name[1],
                                                                                                    self.curr_mp_pi,
                                                                                                    self.old_mp_pi,
                                                                                                    self.mp_adv_ph,
                                                                                                    self.mp_lr_ph)
            else:
                self.tensorboard_keys += ['loss/mp_loss_tr']
                with tf.variable_scope('curr_model_primitive_%s' % c.xml_name[1]):
                    self.next_s_pred_tr = dnn(input=self.s_a_tr, output_size=self.obs_dim, scope='next_s_pred_tr',
                                              n_layers=c.pg_mp_n_layers, size=c.pg_mp_hidden_fc_size, trainable=True)

                    self.mp_loss_tr = vector_l2_loss_tf(self.next_s_pred_tr, self.n_obv_ph, 'mp_loss_tr')
                self.train_mp_op, self.mp_grad_norm_tr = self.train_op(self.mp_loss_tr, self.mp_lr_ph,
                                                                       'curr_model_primitive_%s' % c.xml_name[1])
