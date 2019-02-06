import os

import numpy as np
import tensorflow as tf

from config import c, n
import utils


def calc_lookat(corridors):
    l, t, r, b = zip(* [i['pos'] for i in corridors])
    min_l = min(l)
    max_r = max(r)
    min_b = min(b)
    max_t = max(t)
    return [np.mean([min_l, max_r]), np.mean([min_b, max_t]), 0]


def validate_paths(corridors):
    for i in range(len(corridors)):
        for j in range(i + 1, len(corridors)):
            l1, t1, r1, b1 = corridors[i]['pos']
            l2, t2, r2, b2 = corridors[j]['pos']
            if intersect(l1, l2, r1, r2) and intersect(b1, b2, t1, t2):
                return False
    return True


def intersect(l1, l2, r1, r2):
    if l1 < l2 < r1 or l1 < r2 < r1 or l2 < l1 < r2 or l2 < r1 < r2:
        print('no good...', l1, l2, r1, r2)
        return True
    return False


def main():
    mazes = []
    goals = []
    direction = 2
    for i in range(c.num_mazes):
        valid = False
        while not valid:
            paths = [[0, 0]]
            # NOTE: MAZE
            for j in range(c.num_corr[i]):
                # NOTE: CORRIDOR
                corr_len = np.random.randint(low=c.min_corr_len, high=c.max_corr_len)
                xy = np.zeros(2).tolist()
                if direction < 2:
                    direction = 1 - direction
                else:
                    direction = np.random.randint(low=0, high=2)
                xy[direction] = paths[-1][direction] + corr_len * (
                    np.random.randint(low=0, high=2) * 2 - 1)
                xy[1 - direction] = paths[-1][1 - direction]
                paths.append(xy)
            paths = np.array(paths)
            valid = validate_paths(utils.calc_corridors(paths))
        mazes.append(paths)
        goals.append(paths[-1])

    for i in range(c.num_mazes):
        c.paths = mazes[i]
        c.goal[0] = goals[i][0]
        c.goal[1] = goals[i][1]
        c.max_frames = 1
        c.viewer = True
        c.finalize = False
        c.corridors = utils.calc_corridors(c.paths)
        utils.modify_xml(task_id=None, seed=seed)  # TODO(jkg)
        algo = algo_class(0, False, False, 0)
        algo.record_one_rollout()
        tf.reset_default_graph()

    print(np.array(mazes).tolist())


if __name__ == "__main__":
    module = __import__("algorithms.%s" % c.algo, fromlist=[c.algo])
    algo_class = getattr(module, c.algo)
    for f in os.listdir(os.path.join('results', c.ckpt_path)):
        if '.gif' in f or '.png' or '.mp4' in f:
            os.remove(os.path.join('results', c.ckpt_path, f))
    main()
