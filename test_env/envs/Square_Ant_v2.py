from test_env.envs.Mujoco_Maze import *


class Square_Ant_v2(Mujoco_Maze):
    def __init__(self):
        Mujoco_Maze.__init__(self)

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

        forward_reward = (distbefore - distafter) / self.dt + int(
            distafter < c.success_dist) * c.goal_r
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = c.survival_reward
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[
            2] <= 1.0 and state[5] >= -0.5 and distafter >= c.success_dist
        done = not notdone
        ob = self._get_obs(xposafter, yposafter)
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            laserdist=ob[:4],
            success=int(distafter < c.success_dist),
            orientation=self.sim.data.qpos.flat[3:7])

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
