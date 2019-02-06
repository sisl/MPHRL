import warnings

import gym

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import itertools
import json
import os
import shutil
import time
import traceback

from matplotlib import pyplot as plt
import numpy as np
import ray
import tensorflow as tf

from config import c, n
import utils


def main():
    from config import n
    # NOTE: Plot Config
    if not os.path.exists('results'):
        os.mkdir('results')
    if os.path.exists(os.path.join('results', n(c, seed))):
        print("DELETING...")
        shutil.rmtree(os.path.join('results', n(c, seed)), ignore_errors=True)
    for name in os.listdir('results'):
        name_dir = os.path.join('results', name)
        if os.path.isdir(name_dir):
            events_exist = False
            for number in os.listdir(name_dir):
                if os.path.isdir(os.path.join(name_dir, number)):
                    for f in os.listdir(os.path.join(name_dir, number)):
                        if 'events' in f:
                            events_exist = True
            if not events_exist:
                shutil.rmtree(name_dir)
    os.mkdir(os.path.join('results', n(c, seed)))

    shutil.copyfile('./config.py', os.path.join('results', n(c, seed), 'cfg.py'))

    # NOTE: ALGORITHM
    result, msg = run(np.random.randint(100))
    return 'Job Succeeded!', msg


def run(run_id):
    starting_iter = 0
    task_id = c.task_id
    utils.modify_xml(task_id, seed)
    algo = algo_class(run_id, True, c.restore_model, task_id, seed)
    ray.init(num_cpus=c.num_cores, redirect_output=True)
    algo_threads = ray.remote(algo_class)
    algo.actors = [
        algo_threads.remote(_ + np.random.randint(100) + run_id, False, False, task_id, seed)
        for _ in range(c.num_cores)
    ]

    algo.save_model_op(first=True)
    print('initial eval reward', algo.evaluate_avg_r())

    for iter in range(c.num_batches * c.num_tasks):
        start = time.time()
        algo.env = gym.make('%s%s' % (c.task_id, c.env_name))

        # NOTE: GET EXPERIENCE
        algo.save_model_op()
        ray.get([ac.restore_model_op.remote(algo.run_id) for ac in algo.actors])

        ret = list(
            zip(*ray.get([actor.gen_experiences_op.remote()
                          for actor in algo.actors[:c.num_cpus]])))
        experience = list(itertools.chain(*ret[0]))
        num_traj = sum(ret[1])
        cum_r = sum(ret[2])
        num_succ = sum(ret[3])

        # NOTE: experience is the list of tuples in one rollout
        algo.avg_train_rewards = cum_r / (num_traj + c.e)
        algo.num_traj_per_cpu = num_traj / c.num_cpus
        algo.suc_train = num_succ / (num_traj + c.e)
        algo.num_timesteps = len(experience)
        algo.update_learning_rates_op(iter)

        algo.suc_test = algo.suc_train
        algo.eval_rewards = algo.avg_train_rewards
        algo.suc_test_std = np.std(ret[3])

        if c.gpu:
            with tf.device('/gpu:0'):
                algo.learn(experience)
        else:
            algo.learn(experience)

        # NOTE: LIST
        algo.suc_perc_train_list.append(algo.suc_train)
        algo.eval_r_list.append(algo.eval_rewards)
        algo.suc_perc_test_list.append(algo.suc_test)

        # NOTE: MOV AVG
        algo.suc_perc_train_list = algo.suc_perc_train_list[-c.mov_avg:]
        algo.eval_r_list = algo.eval_r_list[-c.mov_avg:]
        algo.suc_perc_test_list = algo.suc_perc_test_list[-c.mov_avg:]

        algo.mov_suc_train = np.mean(algo.suc_perc_train_list)
        algo.moving_eval_rewards = np.mean(algo.eval_r_list)
        algo.mov_suc_test = np.mean(algo.suc_perc_test_list)
        algo.max_eval_suc_mov = max(algo.mov_suc_test, algo.max_eval_suc_mov)

        algo.seconds = time.time() - start
        print((iter, '%s S' % algo.seconds, algo.eval_rewards, algo.avg_train_rewards, task_id))
        print(n(c, seed))
        if algo.mov_suc_test > algo.max_reward + 0.03:
            algo.save_best_model_op()
            algo.max_reward = algo.mov_suc_test

        # NOTE: LIFELONG LEARNING
        if algo.mov_suc_test >= c.solved_threshold or algo.timesteps_so_far > \
                c.total_ts:
            task_id += 1
            algo.save_best_model_op()
            if task_id == c.num_tasks:
                algo.timesteps_used_list.append(algo.timesteps_so_far / 1e6)
                algo.timesteps_used_list.append(sum(algo.timesteps_used_list))
                algo.fd[algo.timesteps_used_text_ph] = str(algo.timesteps_used_list)
                algo.fd[algo.ts_string_ph] = ' & '.join(
                    [str(round(num, 1)) for num in algo.timesteps_used_list])
                algo.summary()
                break
            algo.suc_perc_test_list = []
            algo.suc_perc_train_list = []
            algo.eval_r_list = []
            utils.modify_xml(task_id, seed)
            algo.reinit(task_id=task_id)
            ray.get([actor.reinit.remote(task_id) for actor in algo.actors])
            algo.iteration_solved = iter - starting_iter
            starting_iter = iter

        algo.memory = 0
        algo.summary()

    return utils.pickle_compatible(algo), algo.fd[algo.ts_string_ph] or 'None'


if __name__ == "__main__":
    from test_env import seed

    print('FINAL SEED', seed)
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

    module = __import__("algorithms.%s" % c.algo, fromlist=[c.algo])
    algo_class = getattr(module, c.algo)
    try:
        subject, msg = main()
        content = n(c, seed) + '\n' + msg
    except:
        subject = 'Job Failed!'
        content = '%s\n%s' % (n(c, seed), traceback.format_exc())
        print(content)
