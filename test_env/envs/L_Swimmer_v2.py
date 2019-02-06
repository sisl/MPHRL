from test_env.envs.Mujoco_Maze import *


class L_Swimmer_v2(Mujoco_Maze):
    def __init__(self):
        Mujoco_Maze.__init__(self)

    def step(self, a):
        if not hasattr(self, 'blocks_id'):
            self.calculate_block_pos()
        ctrl_cost_coeff = 0.0001
        xposbefore = self.get_body_com("mid")[0]
        yposbefore = self.get_body_com("mid")[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("mid")[0]
        yposafter = self.get_body_com("mid")[1]
        reward_fwd = (xposafter - xposbefore + yposafter - yposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs(xposafter, yposafter)
        return ob, reward, False, dict(reward_forward=reward_fwd,
                                       reward_ctrl=reward_ctrl,
                                       reward_contact=0,
                                       reward_survive=0,
                                       laserdist=ob[:4],
                                       orientation=self.sim.data.qpos.flat[3:7])

    def _get_obs(self, xposafter, yposafter):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([self._get_laser_dist(xposafter, yposafter),
                               qpos.flat[2:],
                               qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1,
                                                       size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs(qpos[0], qpos[1])
