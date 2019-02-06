import os
import traceback
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import cv2
import imageio
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from config import c, n

os.environ['NUMEXPR_MAX_THREADS'] = str(c.num_cores)
import numexpr as ne

ne.set_num_threads(c.num_cores)


class RunningMeanStd(object):

    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = tf.get_variable(dtype=tf.float64, shape=shape,
                                    initializer=tf.constant_initializer(0.0), name="runningsum",
                                    trainable=False)
        self._sumsq = tf.get_variable(dtype=tf.float64, shape=shape,
                                      initializer=tf.constant_initializer(epsilon),
                                      name="runningsumsq", trainable=False)
        self._count = tf.get_variable(dtype=tf.float64, shape=(),
                                      initializer=tf.constant_initializer(epsilon), name="count",
                                      trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(
            tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        self.newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        self.assign_add = tf.group([
            tf.assign_add(self._sum, self.newsum),
            tf.assign_add(self._sumsq, self.newsumsq),
            tf.assign_add(self._count, self.newcount)
        ])

    def update(self, x, sess):
        x = x.astype('float64')
        assert x.shape[1] == self.shape, (x.shape[1], self.shape)
        n = int(np.prod(self.shape))
        totalvec = np.concatenate([
            x.sum(axis=0).ravel(),
            np.square(x).sum(axis=0).ravel(),
            np.array([len(x)], dtype='float64')
        ])
        sess.run(self.assign_add, feed_dict={
            self.newsum: totalvec[0:n].reshape(self.shape),
            self.newsumsq: totalvec[n:2 * n].reshape(self.shape),
            self.newcount: totalvec[2 * n]
        })


def generate_seg(dones, rews, baseline):
    seg = {}
    seg['new'] = [1] + dones[:-1]
    seg['rew'] = rews
    seg['vpred'] = baseline
    seg['nextvpred'] = seg['vpred'][-1] * (1 - seg['new'][-1])
    return seg


def add_vtarg_and_adv(seg):
    """
    Compute target value using TD(lambda) estimator and advantage with GAE(
    lambda)
    """
    new = np.append(seg["new"], 0)
    # NOTE: last element is only used for last vtarg, but we
    # already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = seg["rew"].shape[0]
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + c.gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + c.gamma * c.lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def normc_initializer(std=1.0, axis=0):

    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)

    return _initializer


def is_dm_control():
    return c.env_type == 'dm_control'


def is_mujoco():
    return c.env_type == 'mujoco'


def true_model(infos, algo=None):
    if c.xml_name.startswith(('B', 'SP')):
        return np.zeros(shape=len(infos)).astype(np.int64)
    if is_mujoco():
        if c.xml_name.startswith('R'):
            orient = np.stack([i['orientation'] for i in infos], axis=0)
            return np.argmax(orient, axis=1)
        elif c.xml_name.startswith('V'):
            return np.array([np.argmax(i['velocity']) for i in infos])
        elif c.xml_name.startswith(('T', 'Square', 'M', 'C', '2', '5')):
            pos = [i['pos'][:2] for i in infos]
            corridors = algo.env.env.corridors
            if c.xml_name.startswith('C'):
                return np.array([corridor_one_hot_corner(p[0], p[1], corridors) for p in pos])
            elif c.xml_name.startswith('2'):
                return np.array(
                    [corridors[corridor_number(p[0], p[1], corridors)]['model'] % 2 for p in pos])
            elif c.xml_name.startswith('5'):
                bs = len(infos)
                corridor_models = np.array(
                    [corridors[corridor_number(p[0], p[1], corridors)]['model'] for p in pos])
                velocity = np.array([np.argmax(i['velocity']) for i in infos])[:, np.newaxis]
                org = np.zeros(shape=(bs, 4))
                org[np.arange(bs), corridor_models] = 1
                return np.concatenate((org, velocity), axis=1)
            else:
                return np.array(
                    [corridors[corridor_number(p[0], p[1], corridors)]['model'] for p in pos])
        else:
            raise NotImplementedError
    elif is_dm_control():
        if c.xml_name.startswith('V'):
            return np.array([i['stage_obv'][i['box_of_interest']] for i in infos])
        if c.xml_name.startswith('A'):
            return np.array([(i['stage_obv'][i['box_of_interest']] + i['box_of_interest'] * 6)
                             for i in infos])
        if c.xml_name.startswith('2'):
            return np.array([i['box_of_interest'] for i in infos])
        raise NotImplementedError
    else:
        raise NotImplementedError


def threads(num_threads, func, iterable):
    pool = Pool(processes=num_threads)
    ret = pool.map(func=func, iterable=iterable)
    pool.close()
    pool.join()
    return ret


class LinearSchedule(object):

    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        self.epsilon = max(self.eps_end, self.eps_begin - float(t) / self.nsteps *
                           (self.eps_begin - self.eps_end))


def direction(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 == x2:
        # NOTE: N or S
        return 'S' if y1 > y2 else 'N'
    elif y1 == y2:
        # NOTE: E or W
        return 'E' if x2 > x1 else 'W'
    else:
        raise NotImplementedError


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def calc_corridors(paths):
    paths = np.array(paths)
    if len(paths.shape) > 2:
        return
    corridors = []
    for i in range(paths.shape[0] - 1):
        d1 = direction(paths[i], paths[i + 1])
        d2 = direction(paths[i + 1], paths[i + 2]) if i < paths.shape[0] - 2 else 'U'
        d0 = d1 if i == 0 else direction(paths[i - 1], paths[i])

        x1, y1 = paths[i]
        x2, y2 = paths[i + 1]
        # NOTE: Left Top Right Bottom
        cor_w0 = calc_cor_w(d0)
        cor_w1 = calc_cor_w(d1)
        cor_w2 = calc_cor_w(d2) or cor_w1

        e = 2 * int(i == paths.shape[0] - 2) * cor_w1
        pos = {
            'N': (x1 - cor_w1, y2 - cor_w0 + e, x1 + cor_w1, y1 - cor_w2),
            'S': (x1 - cor_w1, y1 + cor_w0, x1 + cor_w1, y2 + cor_w2 - e),
            'E': (x1 - cor_w0, y1 + cor_w1, x2 - cor_w2 + e, y1 - cor_w1),
            'W': (x2 + cor_w0 - e, y1 + cor_w1, x1 + cor_w2, y1 - cor_w1),
        }[d1]
        l, t, r, b = pos
        end = np.array({
            'N': {
                'E': [(l + r) / 2, t],
                'W': [(l + r) / 2, t],
                'U': [(l + r) / 2, t]
            },
            'S': {
                'E': [(l + r) / 2, b],
                'W': [(l + r) / 2, b],
                'U': [(l + r) / 2, t]
            },
            'E': {
                'N': [r, (t + b) / 2],
                'S': [r, (t + b) / 2],
                'U': [r, (t + b) / 2]
            },
            'W': {
                'N': [l, (t + b) / 2],
                'S': [l, (t + b) / 2],
                'U': [r, (t + b) / 2]
            }
        }[d1][d2])

        model = c.dir_to_model[d1]
        corridors.append({
            'pos': pos,
            'd': d1,
            'end': end,
            'length': 0 if i == 0 else np.linalg.norm(end - corridors[i - 1]['end']),
            'model': model
        })
    return corridors


def modify_xml(task_id, seed):
    if c.env_type == 'mujoco':
        for sub_env in range(c.paths[task_id].shape[0]):
            paths = np.array(c.paths[task_id][sub_env])
            newgoal = paths[-1].tolist() + [0.5]
            xml_path = os.path.join(
                os.path.dirname(__file__), 'test_env', 'envs', 'assets',
                c.xml_name[c.xml_name.find('-') + 1:] + '.xml')
            new_xml_path = os.path.join(
                os.path.dirname(__file__), 'test_env', 'envs', 'assets',
                '%s-run-%s.xml' % (n(c, seed), sub_env))

            f = ET.parse(xml_path)
            root = f.getroot()
            goal = root.find("**[@name='pointbody']")
            goal.set('pos', '%s %s %s' % (newgoal[0], newgoal[1], newgoal[2]))
            goal.set('size', str(c.cor_w / 2))
            draw_walls(paths, root.find("worldbody"))
            print(new_xml_path)
            f.write(new_xml_path)


def corridor_one_hot_corner(x, y, corridors):
    ret = np.zeros(c.num_models)
    correct = corridor_number(x, y, corridors)
    if correct > 0:
        l, t, r, b = corridors[correct - 1]['pos']
        dir = corridors[correct - 1]['d']
        if (dir in ['E', 'W'] and b <= y <= t) or \
                (dir in ['N', 'S'] and l <= x <= r):
            ret[corridors[correct]['model']] = 0.5
            ret[corridors[correct - 1]['model']] = 0.5
            return ret
    ret[corridors[correct]['model']] = 1
    return ret


def corridor_number(x, y, corridors):
    for i in range(len(corridors)):
        l, t, r, b = corridors[i]['pos']
        if l <= x <= r and b <= y <= t:
            return i
    # print(x, y, corridors)
    error = 0
    while True:
        error += 0.5
        for i in range(len(corridors)):
            l, t, r, b = corridors[i]['pos']
            if l - error <= x <= r + error and b - error <= y <= t + error:
                return i


def vector_l2_loss_np(pred, gt):
    return np.sum(np.square(gt - pred), axis=1)


def vector_l2_loss_tf(pred, gt, name):
    return tf.reduce_mean(tf.reduce_sum(tf.square(gt - pred), axis=1), name=name)


def wall(tree, pos, size):
    block_id = len(tree.findall('geom'))
    w = ET.SubElement(tree, 'geom')
    w.set('conaffinity', "1")
    w.set('contype', "1")
    w.set('material', "")
    w.set('type', "box")
    w.set('rgba', "0.59 0.31 0.02 1")
    w.set('name', 'block_%s' % block_id)
    w.set('pos', "%s %s %s" % pos)
    w.set('size', "%s %s %s" % size)


def calc_cor_w(d):
    return c.cor_wd[d]


def draw_start_pt_wall(pt1, pt2, pt3, tree):
    d1 = direction(pt1, pt2)
    d2 = direction(pt2, pt3) if pt3 is not None else d1
    cor_w1 = calc_cor_w(d1)
    cor_w2 = calc_cor_w(d2)
    x1, y1 = pt1
    x2, y2 = pt2
    wall_dist1 = cor_w1 + c.wl_w / 2
    wall_dist2 = cor_w2 + c.wl_w / 2

    pos1 = {
        'N': (x1, y1 - wall_dist1, c.wl_h / 2),
        'S': (x1, y1 + wall_dist1, c.wl_h / 2),
        'E': (x1 - wall_dist1, y1, c.wl_h / 2),
        'W': (x1 + wall_dist1, y1, c.wl_h / 2),
    }[d1]
    pos2 = {
        'N': (x2, y2 + wall_dist2, c.wl_h / 2),
        'S': (x2, y2 - wall_dist2, c.wl_h / 2),
        'E': (x2 + wall_dist2, y1, c.wl_h / 2),
        'W': (x2 - wall_dist2, y1, c.wl_h / 2),
    }[d1]
    size = {
        'N': (cor_w1 + c.wl_w, c.wl_w / 2, c.wl_h / 2),
        'S': (cor_w1 + c.wl_w, c.wl_w / 2, c.wl_h / 2),
        'E': (c.wl_w / 2, cor_w1 + c.wl_w, c.wl_h / 2),
        'W': (c.wl_w / 2, cor_w1 + c.wl_w, c.wl_h / 2)
    }[d1]
    wall(tree, pos1, size)
    wall(tree, pos2, size)

    pos1 = {
        'N': (x1 - wall_dist1, (y1 + y2 - cor_w1 - cor_w2) / 2, c.wl_h / 2),
        'S': (x1 - wall_dist1, (y1 + y2 + cor_w1 + cor_w2) / 2, c.wl_h / 2),
        'E': ((x1 + x2 - cor_w1 - cor_w2) / 2, y1 - wall_dist1, c.wl_h / 2),
        'W': ((x1 + x2 + cor_w1 + cor_w2) / 2, y1 - wall_dist1, c.wl_h / 2),
    }[d1]
    pos2 = {
        'N': (x1 + wall_dist1, (y1 + y2 - cor_w1 - cor_w2) / 2, c.wl_h / 2),
        'S': (x1 + wall_dist1, (y1 + y2 + cor_w1 + cor_w2) / 2, c.wl_h / 2),
        'E': ((x1 + x2 - cor_w1 - cor_w2) / 2, y1 + wall_dist1, c.wl_h / 2),
        'W': ((x1 + x2 + cor_w1 + cor_w2) / 2, y1 + wall_dist1, c.wl_h / 2),
    }[d1]
    size = {
        'N': (c.wl_w / 2, abs(y1 - cor_w1 - (y2 - cor_w2)) / 2, c.wl_h / 2),
        'S': (c.wl_w / 2, abs(y1 + cor_w1 - (y2 + cor_w2)) / 2, c.wl_h / 2),
        'E': (abs(x1 - cor_w1 - (x2 - cor_w2)) / 2, c.wl_w / 2, c.wl_h / 2),
        'W': (abs(x1 + cor_w1 - (x2 + cor_w2)) / 2, c.wl_w / 2, c.wl_h / 2)
    }[d1]
    wall(tree, pos1, size)
    wall(tree, pos2, size)


def draw_wall(pt0, pt1, pt2, pt3, tree):
    d0 = direction(pt0, pt1)
    d1 = direction(pt1, pt2)
    d2 = direction(pt2, pt3) if pt3 is not None else d1
    cor_w0 = calc_cor_w(d0)
    cor_w1 = calc_cor_w(d1)
    cor_w2 = calc_cor_w(d2)
    x1, y1 = pt1
    x2, y2 = pt2
    wall_dist0 = cor_w0 + c.wl_w / 2
    wall_dist1 = cor_w1 + c.wl_w / 2
    wall_dist2 = cor_w2 + c.wl_w / 2
    pos1 = {
        'N': (x1, y1 - wall_dist0, c.wl_h / 2),
        'S': (x1, y1 + wall_dist0, c.wl_h / 2),
        'E': (x1 - wall_dist0, y1, c.wl_h / 2),
        'W': (x1 + wall_dist0, y1, c.wl_h / 2),
    }[d1]
    size = {
        'N': (cor_w1, c.wl_w / 2, c.wl_h / 2),
        'S': (cor_w1, c.wl_w / 2, c.wl_h / 2),
        'E': (c.wl_w / 2, cor_w1, c.wl_h / 2),
        'W': (c.wl_w / 2, cor_w1, c.wl_h / 2)
    }[d1]
    wall(tree, pos1, size)
    pos2 = {
        'N': (x2, y2 + wall_dist2, c.wl_h / 2),
        'S': (x2, y2 - wall_dist2, c.wl_h / 2),
        'E': (x2 + wall_dist2, y1, c.wl_h / 2),
        'W': (x2 - wall_dist2, y1, c.wl_h / 2),
    }[d1]
    size = {
        'N': (cor_w1 + c.wl_w, c.wl_w / 2, c.wl_h / 2),
        'S': (cor_w1 + c.wl_w, c.wl_w / 2, c.wl_h / 2),
        'E': (c.wl_w / 2, cor_w1 + c.wl_w, c.wl_h / 2),
        'W': (c.wl_w / 2, cor_w1 + c.wl_w, c.wl_h / 2)
    }[d1]
    wall(tree, pos2, size)

    pos1 = {
        'N': (x1 - wall_dist1, ((y1 + cor_w0 + c.wl_w) + (y2 - cor_w2)) / 2, c.wl_h / 2),
        'S': (x1 - wall_dist1, ((y1 - cor_w0 - c.wl_w) + (y2 + cor_w2)) / 2, c.wl_h / 2),
        'E': (((x1 + cor_w0 + c.wl_w) + (x2 - cor_w2)) / 2, y1 - wall_dist1, c.wl_h / 2),
        'W': (((x1 - cor_w0 - c.wl_w) + (x2 + cor_w2)) / 2, y1 - wall_dist1, c.wl_h / 2),
    }[d1]
    pos2 = {
        'N': (x1 + wall_dist1, ((y1 + cor_w0 + c.wl_w) + (y2 - cor_w2)) / 2, c.wl_h / 2),
        'S': (x1 + wall_dist1, ((y1 - cor_w0 - c.wl_w) + (y2 + cor_w2)) / 2, c.wl_h / 2),
        'E': (((x1 + cor_w0 + c.wl_w) + (x2 - cor_w2)) / 2, y1 + wall_dist1, c.wl_h / 2),
        'W': (((x1 - cor_w0 - c.wl_w) + (x2 + cor_w2)) / 2, y1 + wall_dist1, c.wl_h / 2),
    }[d1]
    # (y1 - cor_w0 - c.wl_w)
    # (y2 + cor_w2)
    size = {
        'N': (c.wl_w / 2, (-(y1 + cor_w0 + c.wl_w) + (y2 - cor_w2)) / 2, c.wl_h / 2),
        'S': (c.wl_w / 2, ((y1 - cor_w0 - c.wl_w) - (y2 + cor_w2)) / 2, c.wl_h / 2),
        'E': ((-(x1 + cor_w0 + c.wl_w) + (x2 - cor_w2)) / 2, c.wl_w / 2, c.wl_h / 2),
        'W': (((x1 - cor_w0 - c.wl_w) - (x2 + cor_w2)) / 2, c.wl_w / 2, c.wl_h / 2)
    }[d1]
    if size[0] > 0 and size[1] > 0 and size[2] > 0:
        wall(tree, pos1, size)
        wall(tree, pos2, size)


def draw_end_pt_wall(pt1, pt2, tree):
    d = direction(pt1, pt2)
    x2, y2 = pt2
    cor_w = calc_cor_w(d)
    wall_dist = cor_w + c.wl_w / 2
    pos1 = {
        'N': (x2 - wall_dist, y2, c.wl_h / 2),
        'S': (x2 - wall_dist, y2, c.wl_h / 2),
        'E': (x2, y2 - wall_dist, c.wl_h / 2),
        'W': (x2, y2 - wall_dist, c.wl_h / 2),
    }[d]
    pos2 = {
        'N': (x2 + wall_dist, y2, c.wl_h / 2),
        'S': (x2 + wall_dist, y2, c.wl_h / 2),
        'E': (x2, y2 + wall_dist, c.wl_h / 2),
        'W': (x2, y2 + wall_dist, c.wl_h / 2),
    }[d]
    size = {
        'N': (c.wl_w / 2, cor_w, c.wl_h / 2),
        'S': (c.wl_w / 2, cor_w, c.wl_h / 2),
        'E': (cor_w, c.wl_w / 2, c.wl_h / 2),
        'W': (cor_w, c.wl_w / 2, c.wl_h / 2)
    }[d]
    wall(tree, pos1, size)
    wall(tree, pos2, size)


def draw_walls(paths, tree):
    paths = np.array(paths)
    num_pts = paths.shape[0]
    draw_start_pt_wall(paths[0], paths[1], paths[2] if len(paths) > 2 else None, tree)
    for i in range(1, num_pts - 1):
        if i == num_pts - 2:
            draw_wall(paths[i - 1], paths[i], paths[i + 1], None, tree)
        else:
            draw_wall(paths[i - 1], paths[i], paths[i + 1], paths[i + 2], tree)
    draw_end_pt_wall(paths[-2], paths[-1], tree)


def save_video(frames, filename, fps):
    png_dir = os.path.expanduser("~/Desktop/images/")
    t = 0
    l = len(frames)
    gif = []
    for frame in frames:
        t += 1
        print(t, l, end='\r')
        if c.domain == 'maze':
            img, x, y, dx, dy, reward, info = frame
            belief_str = 'belief: '
            for value in info['belief']:
                belief_str += str(round(value, 1)) + ','
        else:
            img, reward, info = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.flip(frame[0], 0)

        if c.gif_format == 'mp4':
            if c.domain == 'maze':
                x = 10
                y_loc = 10
                box_size = 40
                y_inc = int(box_size * 2.8 // 4)
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=c.gold, thickness=cv2.FILLED)
                img = cv2.putText(img, text='E', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='N', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 255, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='W', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='S', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad + box_size // 2
                if c.num_models > 1:
                    img = cv2.putText(img, text=str(round(info['ns_logprob'][0], 2)),
                                      org=(10, y_loc + y_inc), fontFace=font, fontScale=c.fontscale,
                                      thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                    img = cv2.putText(img, text=str(round(info['ns_logprob'][1], 2)),
                                      org=(10,
                                           y_loc + 2 * y_inc), fontFace=font, fontScale=c.fontscale,
                                      thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                    img = cv2.putText(img, text=str(round(info['ns_logprob'][2], 2)),
                                      org=(10,
                                           y_loc + 3 * y_inc), fontFace=font, fontScale=c.fontscale,
                                      thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                    img = cv2.putText(img, text=str(round(info['ns_logprob'][3], 2)),
                                      org=(10,
                                           y_loc + 4 * y_inc), fontFace=font, fontScale=c.fontscale,
                                      thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                    img = cv2.putText(img, text=str(np.round(info['pred_laser'][0][0], 1)),
                                      org=(x + box_size // 4,
                                           y_loc + y_inc - 2), fontFace=font, fontScale=c.fontscale,
                                      thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                    img = cv2.putText(img, text=str(np.round(info['pred_laser'][1][0], 1)),
                                      org=(x + box_size // 4, y_loc + 2 * y_inc - 2), fontFace=font,
                                      fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                      color=(255, 255, 255))
                    img = cv2.putText(img, text=str(np.round(info['pred_laser'][2][0], 1)),
                                      org=(x + box_size // 4, y_loc + 3 * y_inc - 2), fontFace=font,
                                      fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                      color=(255, 255, 255))
                    img = cv2.putText(img, text=str(np.round(info['pred_laser'][3][0], 1)),
                                      org=(x + box_size // 4, y_loc + 4 * y_inc - 2), fontFace=font,
                                      fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                      color=(255, 255, 255))
                    img = cv2.putText(img, text=str(np.round(info['pred_laser'][3][1], 1)),
                                      org=(x + box_size // 4, y_loc + 5 * y_inc - 2), fontFace=font,
                                      fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                      color=(255, 255, 255))
            else:
                x = 10
                y_loc = 310
                img = cv2.putText(img, text='R: Reach', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc += 15
                img = cv2.putText(img, text='L: Lower to', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc += 15
                img = cv2.putText(img, text='G: Grasp', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc += 15
                img = cv2.putText(img, text='P: Pick up', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc += 15
                img = cv2.putText(img, text='C: Carry', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc += 15
                img = cv2.putText(img, text='D: Drop', org=(x, y_loc), fontFace=font,
                                  fontScale=c.fontscale, thickness=c.thickness + 1, lineType=5,
                                  color=(255, 255, 255))
                y_loc = 400
                box_size = 32
                y_inc = int(box_size * 2.8 // 4)
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(255, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='R', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 255, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='L', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 0, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='G', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=c.gold, thickness=cv2.FILLED)
                img = cv2.putText(img, text='P', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(255, 0, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='C', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(0, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='1', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='D', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(255, 0, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='R', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 255, 0), thickness=cv2.FILLED)
                img = cv2.putText(img, text='L', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 0, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='G', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=c.gold, thickness=cv2.FILLED)
                img = cv2.putText(img, text='P', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(255, 0, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='C', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))
                x += c.line_pad // 2 + box_size
                img = cv2.rectangle(img, pt1=(x, y_loc), pt2=(x + box_size, y_loc + box_size),
                                    color=(255, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='2', org=(x + box_size // 4, y_loc + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(0, 0, 0))
                img = cv2.rectangle(img, pt1=(x, y_loc + box_size), pt2=(x + box_size,
                                                                         y_loc + 2 * box_size),
                                    color=(0, 255, 255), thickness=cv2.FILLED)
                img = cv2.putText(img, text='D', org=(x + box_size // 4, y_loc + box_size + y_inc),
                                  fontFace=font, fontScale=c.fontscale + 0.4,
                                  thickness=c.thickness + 1, lineType=5, color=(255, 255, 255))

                gif.append(img)
        elif c.gif_format == 'png':
            base = os.path.basename(filename)
            path = png_dir + '%s.png' % (base[:base.find('-')])
            print(path)
            imageio.imwrite(path, img)

        else:
            y_loc = 20

            img = cv2.copyMakeBorder(img, 0, 0, 200, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.putText(img=img, text='Frame: %s' % round(t, 2), org=(2, y_loc),
                              fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                              thickness=c.thickness, lineType=5)
            y_loc += c.line_pad
            img = cv2.putText(img=img, text='Reward: %s' % round(reward, 2), org=(2, y_loc),
                              fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                              thickness=c.thickness, lineType=5)
            y_loc += c.line_pad
            img = cv2.putText(img=img, text='Dist: %s' % round(info['dist'], 2), org=(2, y_loc),
                              fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                              thickness=c.thickness, lineType=5)
            if c.domain == 'maze':
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='X: %s' % round(x, 2), org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Y: %s' % round(y, 2), org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='subpolicy: %s' % np.argmax(info['belief']),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text=belief_str, org=(2, y_loc), fontFace=font,
                                  fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='dX: %s' % round(dx, 2), org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='dY: %s' % round(dy, 2), org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Forward: %s' % round(info['reward_forward'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Control: %s' % round(info['reward_ctrl'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Contact: %s' % round(info['reward_contact'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='xposlaser: %s' % round(info['laserdist'][0], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='xneglaser: %s' % round(info['laserdist'][1], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='yposlaser: %s' % round(info['laserdist'][2], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='yneglaser: %s' % round(info['laserdist'][3], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Correct model: %s' % info['correct_model'],
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
            elif c.domain == 'manipulator':
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='T: %s' % np.round(info['goal'], 1), org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Ball: %s' % np.round(info['ball'], 1),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Pos: %s' % np.round(info['position'], 1),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Hand: %s' % np.round(info['grasp'], 4),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img,
                                  text='dist_to_obj: %s' % round(info['dist_to_object'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img,
                                  text='dist_to_tar: %s' % round(info['dist_to_target'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='success: %s' % info['success'], org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Last: %s' % info['last'], org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
            elif c.domain == 'stacker':
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='target: %s' % np.round(info['target'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='box_of_int: %s' % info['box_of_interest'],
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img,
                                  text='box_of_ints: %s' % np.round(info['box_of_interests'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='dists: %s' % np.round(info['dists'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad

                img = cv2.putText(img=img, text='Stage: %s' % info['stage_obv'], org=(2, y_loc),
                                  fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                                  thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Hand: %s' % np.round(info['hand'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='Grasp: %s' % np.round(info['grasp'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='touch: %s' % np.round(info['touch'], 2),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='box_size: %s' % round(info['box_size'], 4),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(
                    img=img,
                    text='grasp_to_abo_tar_dist: %s' % np.round(info['grasp_to_abo_tar_dist'], 2),
                    org=(2, y_loc), fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                    thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(
                    img=img, text='target to drop: %s' % np.round(c.target_to_drop[c.task_id], 2),
                    org=(2, y_loc), fontFace=font, fontScale=c.fontscale, color=(255, 255, 255),
                    thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='finger_dist: %s' % round(info['finger_dist'], 5),
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)
                y_loc += c.line_pad
                img = cv2.putText(img=img, text='correct_model: %s' % info['correct_model'],
                                  org=(2, y_loc), fontFace=font, fontScale=c.fontscale,
                                  color=(255, 255, 255), thickness=c.thickness, lineType=5)

            gif.append(img)

    print('Saving image/gif...')
    if c.gif_format in ('gif', 'mp4'):
        imageio.mimsave(uri=filename, ims=np.array(gif), fps=fps)
    else:
        base = os.path.basename(filename)
        path = png_dir + '%s.png' % (base[:base.find('-')])
        print(path)
        imageio.imwrite(path, img)
    print('Saved')


def calc_true_model_index(info, corridor):
    x, y = info['pos'][:2]
    return corridor[corridor_number(x, y, corridor)]['model']


def func_calc_true_model_index(_):
    try:
        return calc_true_model_index(*_)
    except:
        traceback.print_exc()
        raise Exception


def mujoco_transition_model(env):
    return


def dnn(input, output_size, scope, n_layers, size, output_activation=None, trainable=True,
        hid_init=None, final_init=None):
    out = input
    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = layers.fully_connected(out, size, normalizer_fn=None,
                                         activation_fn=c.activation_fn, reuse=False,
                                         trainable=trainable, weights_initializer=hid_init)

        out = layers.fully_connected(out, output_size, normalizer_fn=None,
                                     activation_fn=output_activation, reuse=False,
                                     trainable=trainable, weights_initializer=final_init)

    return out


def dnn_mp(input, output_size, scope, n_layers, size, trainable, hid_init, final_init):
    out = input

    activation_fn = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'None': None}[c.activation_fn_str]

    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = layers.fully_connected(out, size, normalizer_fn=None, activation_fn=activation_fn,
                                         reuse=False, trainable=trainable,
                                         weights_initializer=hid_init)

        out = layers.fully_connected(out, output_size, normalizer_fn=None, activation_fn=None,
                                     reuse=False, trainable=trainable,
                                     weights_initializer=final_init)

    return out


def pickle_compatible(obj):
    newobj = dict()
    obj_dict = obj.__dict__
    keys = obj_dict.keys()
    for key in keys:
        if key == 'env' or isinstance(obj_dict[key], (int, float, complex, tuple, list, dict, set,
                                                      str, np.ndarray)):
            newobj[key] = obj_dict[key]
    return newobj


def experience_to_traj(experience):
    trajectories = []
    trajectory = []
    for example in experience:
        time_step, s, a, r, next_s, done = example[:6]
        trajectory.append(example)
        if done:
            trajectories.append(trajectory)
            trajectory = []
    if not done:
        trajectories.append(trajectory)
    return trajectories


def get_returns(trajectories):
    all_returns = []
    for traj in trajectories:
        rewards = [i[3] for i in traj]
        ret = []
        T = len(rewards)
        for t in range(T - 1, -1, -1):
            r = rewards[t] + (ret[-1] * c.gamma if t < T - 1 else 0)
            ret.append(r)
        ret = ret[::-1]
        all_returns.append(ret)

    return np.concatenate(all_returns)
