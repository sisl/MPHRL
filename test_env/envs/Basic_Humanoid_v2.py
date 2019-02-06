from test_env.envs.Mujoco_Maze import *


class Basic_Humanoid_v2(Mujoco_Maze):
    def __init__(self, paths):
        Mujoco_Maze.__init__(self, paths)

    def _get_obs(self, x, y):
        data = self.sim.data
        return np.concatenate([self._get_laser_dist(x, y),
                               self.goal_direction(x, y),
                               data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        if not hasattr(self, 'blocks_id'):
            self.calculate_block_pos()
        xposbefore, yposbefore = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        xposafter, yposafter = mass_center(self.model, self.sim)
        alive_bonus = c.survival_reward
        data = self.sim.data
        distbefore = self.distance_to_goal(xposbefore, yposbefore)
        distafter = self.distance_to_goal(xposafter, yposafter)
        lin_vel_cost = 0.25 / self.model.opt.timestep * (
                distbefore - distafter) + int(
            distafter < c.success_dist) * c.goal_r
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool(
            (qpos[2] < 1) or (qpos[2] > 2.5) or (distafter < c.success_dist))
        ob = self._get_obs(xposafter, yposafter)
        return ob, reward, done, dict(
            reward_forward=lin_vel_cost,
            reward_ctrl=-quad_ctrl_cost,
            reward_survive=alive_bonus,
            reward_contact=-quad_impact_cost,
            laserdist=ob[:4],
            dist=distafter,
            pos=[xposbefore, yposbefore],
            orientation=self.sim.data.qpos.flat[3:7],
            success=int(distafter < c.success_dist))

    def reset_model(self):
        const = 0.01
        qpos = self.init_qpos + self.np_random.uniform(low=-const, high=const,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-const, high=const,
                                                       size=self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs(qpos[0], qpos[1])
