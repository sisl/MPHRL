from test_env.envs.Mujoco_Maze import *


class Basic_Ant_v2(Mujoco_Maze):
    def __init__(self, paths, seed):
        Mujoco_Maze.__init__(self, paths, seed)
        self.ant_bodies = ['torso_geom', 'left_leg_geom', 'left_ankle_geom',
                           'aux_1_geom', 'aux_2_geom', 'right_leg_geom', 'right_ankle_geom',
                           'aux_3_geom', 'back_leg_geom', 'third_ankle_geom',
                           'aux_4_geom', 'rightback_leg_geom', 'fourth_ankle_geom']
        self.geom_ids = [self.model.geom_name2id(body) for body in
                         self.ant_bodies]
        self.model_to_color = [[0.8, 0.6, 0.4, 1], [1, 0, 0, 1], [0, 1, 0, 1],
                               [0, 0, 1, 1]]

    def get_pos(self):
        return [self.get_body_com("torso")[0], self.get_body_com("torso")[1]]

    def step(self, a):
        if not hasattr(self, 'blocks_id'):
            self.calculate_block_pos()
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        distbefore = self.distance_to_goal(xposbefore, yposbefore)
        distafter = self.distance_to_goal(xposafter, yposafter)

        forward_reward = c.forward_mult * (distbefore - distafter) / (
            1 if c.pos_r else self.dt)
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = c.survival_reward
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0 and state[
            5] >= -0.5 and distafter >= c.success_dist
        done = not notdone
        if distafter >= c.success_dist and done:
            reward -= c.goal_r

        ob = self._get_obs(xposafter, yposafter)

        info = dict(velocity=[abs((xposafter - xposbefore) / self.dt), abs((yposafter - yposbefore) / self.dt)],
                    reward_forward=forward_reward,
                    reward_ctrl=-ctrl_cost,
                    reward_contact=-contact_cost,
                    reward_survive=survive_reward,
                    laserdist=ob[:4],
                    dist=distafter,
                    pos=[xposbefore, yposbefore],
                    success=int(distafter < c.success_dist),
                    orientation=self.sim.data.qpos.flat[3:7])

        return ob, reward, done, info

    def _get_obs(self, x, y):
        return np.concatenate([
            self._get_laser_dist(x, y),
            self.goal_direction(x, y),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq,
                                                       low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs(qpos[0], qpos[1])
