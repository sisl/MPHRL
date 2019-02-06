import copy
import math
import multiprocessing as mp
import warnings as w

import gym
from gym import logger

from utils import *


class algorithm(object):

    def reinit(self, task_id):
        raise NotImplementedError

    def __init__(self, run_id, main_thread, restore, task_id, seed, record=False):
        print('seed', seed)
        self.name = n(c, seed)
        if not c.validate:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            logger.set_level(logger.ERROR)
            tf.logging.set_verbosity(tf.logging.WARN)
            w.filterwarnings("ignore", message="numpy.dtype size changed")
            w.filterwarnings("ignore", message="numpy.ufunc size changed")
        self.run_id = run_id
        self.actors = None
        self.fd = {}
        self.iteration_solved = 0
        self.main = main_thread
        self.t = -1
        self.timesteps_so_far = float(c.prev_timestep)
        self.task_id = task_id
        if self.is_dm_control() and 'SUNet' not in c.hostname:
            os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
        self.env = gym.make('%s%s' % (task_id, c.env_name))
        self.env.seed(self.run_id)
        self.num_timesteps = 0
        self.max_reward = -math.inf
        self.eval_r_list = []
        self.suc_perc_test_list = []
        self.suc_perc_train_list = []
        self.avg_train_rewards = -math.inf
        self.max_eval_suc_mov = -math.inf

        # NOTE: tensorboard plots
        self.tensorboard_keys = [
            'sys/seconds', 'sys/num_traj_per_cpu', 'sys/memory', 'lr/eps',
            'rewards/avg_train_rewards', 'rewards/eval_rewards', 'rewards/moving_eval_rewards',
            'rewards/max_reward', 'timesteps/eval_timesteps', 'percentage/suc_train',
            'percentage/suc_test', 'percentage/max_eval_suc_mov', 'percentage/suc_test_std',
            'percentage/mov_suc_test', 'percentage/mov_suc_train', 'percentage/avg_sub_env_index'
        ]

        self.custom_init_op()
        if self.main:
            self.egreedy_schedule = LinearSchedule(eps_begin=c.eps_begin, eps_end=c.eps_end,
                                                   nsteps=c.decay_steps)
            self.alpha_schedule = LinearSchedule(eps_begin=c.alpha_begin, eps_end=c.alpha_end,
                                                 nsteps=c.decay_steps)
            self.summary_phs = {}
            for tb_key in self.tensorboard_keys:
                key = tb_key.split('/')[-1]
                if tb_key.endswith('_tr'):
                    tf.summary.scalar(tb_key[:-3], getattr(self, key))
                else:
                    self.summary_phs[tb_key] = tf.placeholder(tf.float64, shape=(), name=tb_key)
                    tf.summary.scalar(tb_key, self.summary_phs[tb_key])

            self.summary_op = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(logdir=os.path.join(
                'results', self.name, str(self.run_id)), graph=tf.get_default_graph())

        config = tf.ConfigProto(inter_op_parallelism_threads=mp.cpu_count() if self.main else
                                c.num_models, intra_op_parallelism_threads=mp.cpu_count()
                                if self.main else c.num_models)

        self.sess = tf.Session(config=config)
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()
        if c.use_model_primitive:
            self.savers = []
            for i in range(c.num_models):
                self.savers.append(
                    tf.train.Saver(
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope='curr_model_primitive_%s' % i)))
                self.restore_model_primitive(i)
        if c.finalize:
            self.sess.graph.finalize()
        if restore:
            self.restore_model_best_op()
            # self.restore_model_best_curr_op()

    def evaluate_avg_r(self):
        raise NotImplementedError

    def transfer_learning(self):
        return self.task_id > 0 and c.transfer

    def evaluate_one_rollout(self, record, sub=None):
        self.random_seed()
        s = self.env.reset()
        suc, t, cum_r, surv_r, forw_r, cont_r, ctrl_r = [0] * 7
        frms = []
        while True:
            t += 1
            if isinstance(s, (int, float, np.int64)):
                s = [s]
            if self.is_mujoco():
                a, mp_pred = self.get_action(s, sub=sub, pos=self.env.env.get_pos())
            else:
                a = self.get_action(s, sub=sub)
            s, r, done, info = self.env.step(a)
            suc += info['success']
            surv_r += info['reward_survive']
            forw_r += info['reward_forward']
            ctrl_r += info['reward_ctrl']
            cont_r += info['reward_contact']
            cum_r += r
            if c.record and record:
                print('CORE %s evaluating...' % self.run_id, end='\r')
                img = self.env.env.sim.render(c.gif_s, c.gif_s)
                x, y = self.env.env.sim.data.qpos.flat[:2]
                dx, dy = self.env.env.sim.data.qvel.flat[:2]
                frms.append((img, x, y, dx, dy, cum_r))
            if done:
                break
        self.env.reset()
        return cum_r, frms, suc, t, forw_r, surv_r, ctrl_r, cont_r, info

    def calc_true_model_index(self, info):
        return true_model(info, self)

    def record_one_rollout(self, sub=None):
        task = self.task_id
        self.random_seed()
        if c.viewer and self.is_mujoco():
            self.env.env._get_viewer('rgb_array')
            print('lookat', self.env.env.lookat)
            self.env.env.viewer.cam.lookat[:3] = self.env.env.lookat
        s = self.env.reset()
        t, cum_r = 0, 0
        frames = []
        while True:
            t += 1
            if isinstance(s, (int, float, np.int64)):
                s = [s]
            a, belief = self.get_action(s, view_master=True, sub=sub)
            if self.is_mujoco():
                next_s, r, done, info = self.env.step(a)
                self.pred_gt_obvs = self.calc_post_belief([s], [a], [next_s], [info])
                info['ns_logprob'] = self.sess.run(
                    self.ns_logprobs_tr, feed_dict={self.pred_gt_obvs_ph: self.pred_gt_obvs})[0]
                info['ns_logprob'] -= info['ns_logprob'].max()
                info['pred_laser'] = self.pred_gt_obvs[0, :, :, :4]
                s = next_s
            else:
                s, r, done, info = self.env.step(a, np.argmax(belief))
            if self.is_mujoco():
                info['belief'] = belief
                info['correct_model'] = true_model([info], algo=self)[0]
                if c.change_color:
                    for id in self.env.env.geom_ids:
                        color = self.env.env.model_to_color[np.argmax(belief)]
                        self.env.env.model.geom_rgba[id, :] = color

            cum_r += r
            if c.viewer:
                if self.is_mujoco():
                    x, y = self.env.env.viewer.sim.data.qpos.flat[:2]
                    dx, dy = self.env.env.viewer.sim.data.qvel.flat[:2]
                    img = self.env.render(mode='rgb_array', width=c.gif_s, height=c.gif_s)[::-1]
                    frames.append((img, x, y, dx, dy, cum_r, info))
                else:
                    img = self.env.render(mode='rgb_array')[::-1]
                    frames.append((img, cum_r, info))
            print('cum reward: %s' % cum_r, end='\r')
            if done or t >= c.max_frames:
                print(done)
                break

        cum_r = round(cum_r, 2)
        file = 'task=%s-r=%s-suc=%s-sub=%s.%s' % (task, cum_r, bool(info['success']), sub,
                                                  c.gif_format)
        if c.viewer or (c.view_sub or info['success']):
            print('RECORDING...')
            record_path = os.path.join('results', c.ckpt_path)
            save_video(frames, os.path.join(record_path, file), 20)
        self.env.reset()
        print('DONE %s' % file)
        return cum_r, bool(info['success'])

    def random_seed(self):
        self.env.seed(np.random.randint(10000) + self.run_id)

    def is_mujoco(self):
        return c.env_type == 'mujoco'

    def is_dm_control(self):
        return c.env_type == 'dm_control'

    def gen_experiences_op(self):
        self.random_seed()
        self.env = gym.make('%s%s' % (self.task_id, c.env_name))
        s = self.env.reset()
        experience, traj, traj_r = [], [], []
        cum_r, num_traj, num_succ, curr_t_steps = [0] * 4
        for _ in range(c.bs_per_cpu):
            if _ % 500 == 0:
                print('Progress: %s / %s, %s' % (_, c.bs_per_cpu, self.run_id), end='\r')
            if isinstance(s, (int, float, np.int64)):
                s = [s]
            if self.is_mujoco():
                a, mp_act = self.get_action(s, pos=self.env.env.get_pos())
            else:
                a, mp_act = self.get_action(s), None
            next_s, r, done, info = self.env.step(a)
            info['mp_act'] = mp_act
            if self.is_mujoco():
                info['sub_env_index'] = self.env.env.sub_env_index
            num_succ += info['success']
            traj.append([curr_t_steps, s, a, r, next_s, done, info])
            curr_t_steps += 1
            cum_r += r

            if done:
                experience += traj
                traj = []
                s = self.env.reset()
                num_traj += 1
                traj_r.append(cum_r)
                curr_t_steps, cum_r = 0, 0
            else:
                s = next_s
        if traj:
            experience += traj
        s = self.env.reset()
        if self.is_mujoco():
            assert np.allclose(
                s[self.env.env.dx:self.env.env.dx + 2] - self.env.env.sim.data.qvel.flat[:2], 0)
        return experience, num_traj, sum(traj_r), num_succ

    def save_model_op(self, first=False):
        if not first and not os.path.exists(os.path.join('results', self.name, str(self.run_id))):
            raise NotADirectoryError

        self.saver.save(self.sess,
                        os.path.join('results', self.name, str(self.run_id), 'model.ckpt'))
        print('model saved', self.run_id)

    def save_best_model_op(self):
        self.saver.save(self.sess,
                        os.path.join('results', self.name,
                                     str(self.run_id), 'task=%s-model-best.ckpt' % self.task_id))
        print('best model saved', self.run_id, self.task_id)

    def restore_model_primitives(self):
        if c.use_model_primitive:
            for i in range(c.num_models):
                self.restore_model_primitive(i)

    def restore_model_primitive(self, mp_index):
        path = c.model_primitive_checkpoints[mp_index]
        saver = self.savers[mp_index]
        if not os.path.exists(os.path.join('results', path)):
            raise NotADirectoryError

        saver.restore(self.sess, os.path.join('results', path, 'model.ckpt'))
        print('model primitive restored', path)

    def restore_model_op(self, run_id):
        if not os.path.exists(os.path.join('results', self.name, str(run_id))):
            raise NotADirectoryError

        self.saver.restore(self.sess, os.path.join('results', self.name, str(run_id), 'model.ckpt'))
        print('model restored', run_id, end='\r')

    def restore_model_best_op(self, task_id=c.task_id):
        self.saver.restore(
            self.sess, os.path.join('results', c.ckpt_path, 'task=%s-model-best.ckpt') % task_id)
        print('best model restored', c.ckpt_path, end='\r')

    def restore_model_best_curr_op(self):
        self.saver.restore(self.sess, os.path.join('results', c.ckpt_path, 'model.ckpt'))
        print('model restored', c.ckpt_path, end='\r')

    def custom_init_op(self):
        pass

    def summary(self):
        if not os.path.exists(os.path.join('results', self.name, str(self.run_id))):
            raise NotADirectoryError

        if hasattr(self, 'sess'):
            for key in self.summary_phs:
                self.fd[self.summary_phs[key]] = getattr(self, key.split('/')[-1])
            summary = self.sess.run(self.summary_op, feed_dict=self.fd)
            self.file_writer.add_summary(summary=summary, global_step=self.t + 1)

    def get_action(self, s, view_master=False, sub=None):
        raise NotImplementedError("Should have implemented this")

    def learn(self, experience):
        raise NotImplementedError("Should have implemented this")

    def update_learning_rates_op(self, t):
        self.t = t
        self.egreedy_schedule.update(t=t)
        self.alpha_schedule.update(t=t)
        self.eps = float(self.egreedy_schedule.epsilon)
        self.alpha = float(self.alpha_schedule.epsilon)
