import numpy as np

from test_env.envs.Square_Ant_v2 import Square_Ant_v2


class SquareNoDist_Ant_v2(Square_Ant_v2):
    def __init__(self):
        Square_Ant_v2.__init__(self)

    def _get_obs(self):
        x, y = self.xposafter, self.yposafter
        return np.concatenate([
            self._get_laser_dist(x, y),  # NOTE: laser dist_obv, 4
            self.sim.data.qpos.flat[2:],  # NOTE: partial pos, 111
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
