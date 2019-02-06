import math
import os
import random

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from generate_maze import calc_lookat
from run import c, n
from utils import calc_corridors, corridor_number


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[:2]


class Mujoco_Maze(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, paths, seed):
        num_sub_envs = len(paths)
        self.sub_env_index = random.randint(0, num_sub_envs - 1)
        self.paths = np.array(paths[self.sub_env_index])
        self.corridors = calc_corridors(self.paths)
        self.goal = self.paths[-1].tolist() + [0.5]
        self.lookat = calc_lookat(corridors=self.corridors)

        full_path = os.path.join(os.path.dirname(__file__), 'assets',
                                 '%s-run-%s.xml' % (n(c, seed), self.sub_env_index))
        mujoco_env.MujocoEnv.__init__(self, full_path, 5)
        utils.EzPickle.__init__(self)
        self.dx = 4 + 2 + len(self.sim.data.qpos.flat) - 2

    def step(self, a):
        raise NotImplementedError

    def goal_direction(self, x, y):
        dist = np.abs(np.array([x - self.goal[0], y - self.goal[1]]))
        # dist = np.array([1, 1])
        return dist / np.linalg.norm(dist)

    def reset_model(self):
        raise NotImplementedError

    def _get_laser_dist(self, x, y):
        in_y_range = (self.bottom <= y) * (y <= self.top)
        in_x_range = (self.left <= x) * (x <= self.right)
        blk_on_right = (self.left >= x) * in_y_range
        blk_on_left = (self.left < x) * in_y_range
        blk_on_top = (self.bottom >= y) * in_x_range
        blk_at_bottom = (self.bottom < y) * in_x_range
        xposlaser = np.min(blk_on_right * (self.left - x) + 1e8 * (1 - blk_on_right))
        xneglaser = np.min(blk_on_left * (x - self.right) + 1e8 * (1 - blk_on_left))
        yposlaser = np.min(blk_on_top * (self.bottom - y) + 1e8 * (1 - blk_on_top))
        yneglaser = np.min(blk_at_bottom * (y - self.top) + 1e8 * (1 - blk_at_bottom))
        ret = np.array([xposlaser, xneglaser, yposlaser, yneglaser])

        if c.validate:
            xposlaser, xneglaser, yposlaser, yneglaser = 1e8, 1e8, 1e8, 1e8
            for i in range(len(self.blocks_id)):
                left, right, bottom, top = self.left[i], self.right[i], \
                                           self.bottom[i], self.top[i]
                if bottom <= y <= top:
                    if left >= x:
                        xposlaser = min(xposlaser, left - x)
                    else:
                        xneglaser = min(xneglaser, x - right)
                elif left <= x <= right:
                    if bottom >= y:
                        yposlaser = min(yposlaser, bottom - y)
                    else:
                        yneglaser = min(yneglaser, y - top)
            ret2 = np.array([xposlaser, xneglaser, yposlaser, yneglaser])

            assert np.allclose(ret - ret2, 0)

        if sum(ret) == math.inf:
            for i in range(4):
                if ret[i] == math.inf:
                    ret[i] = 10
        return ret

    def distance_to_goal(self, x, y):
        corridor = corridor_number(x, y, self.corridors)
        goal_corridor = corridor_number(self.goal[0], self.goal[1],
                                        self.corridors)
        if corridor == goal_corridor:
            return np.linalg.norm([x - self.goal[0], y - self.goal[1]])
        elif corridor < goal_corridor:
            start_corr = corridor
            end_corr = goal_corridor
            start_pos = np.array([x, y])
            end_pos = self.goal[:2]
        else:
            start_corr = goal_corridor
            end_corr = corridor
            start_pos = self.goal[:2]
            end_pos = np.array([x, y])

        dist = np.linalg.norm(start_pos - self.corridors[start_corr]['end'])
        start_corr += 1
        while start_corr < end_corr:
            dist += self.corridors[start_corr]['length']
            start_corr += 1
        dist += np.linalg.norm(self.corridors[start_corr - 1]['end'] - end_pos)
        return dist

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * c.viewer_dist

    def calculate_block_pos(self):
        self.blocks_id = [i for i in range(len(self.model.geom_names)) if 'block' in self.model.geom_names[i]]
        self.block_pos = np.array([self.model.geom_pos[i] for i in self.blocks_id])
        self.block_size = np.array([self.model.geom_size[i] for i in self.blocks_id])
        self.left = self.block_pos[:, 0] - self.block_size[:, 0]
        self.right = self.block_pos[:, 0] + self.block_size[:, 0]
        self.bottom = self.block_pos[:, 1] - self.block_size[:, 1]
        self.top = self.block_pos[:, 1] + self.block_size[:, 1]
