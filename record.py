import argparse
import os

import ray

from config import c
import utils

if c.env_type == 'mujoco':
    c.corridors = utils.calc_corridors(c.paths)


def main(algo_class):
    for f in os.listdir(os.path.join('results', c.ckpt_path)):
        if '.gif' in f or '.png' in f or '.mp4' in f:
            os.remove(os.path.join('results', c.ckpt_path, f))

    num_cpus = c.num_models + 1 if c.viewer else c.num_eval
    ray.init(num_cpus=num_cpus, redirect_output=True)

    utils.modify_xml(task_id=c.task_id, seed=seed)
    algo = algo_class(0, False, False, c.task_id, seed, record=True)
    if c.max_frames == 1:
        for task_id in range(c.task_id, c.num_tasks):
            utils.modify_xml(task_id=task_id, seed=seed)
            algo.reinit(task_id)
            algo.record_one_rollout()
    else:
        if c.view_sub:
            algo.restore_model_best_op(c.task_id)
            for i in range(c.num_models):
                algo.record_one_rollout(i)

        elif c.gif_format == 'png':
            algo.restore_model_best_op(c.task_id)
            algo.record_one_rollout()
        else:
            algo.restore_model_best_curr_op()
            algo.restore_model_primitives()
            algo.record_one_rollout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()
    print('SEED', args.seed)
    seed = args.seed

    module = __import__("algorithms.%s" % c.algo, fromlist=[c.algo])
    algo_class = getattr(module, c.algo)
    if c.record_all_lifelong_tasks:
        main(algo_class)
    else:
        main(algo_class)
