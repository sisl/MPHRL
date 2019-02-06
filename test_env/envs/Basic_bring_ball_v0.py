from gym_dmcontrol import *


class Basic_bring_ball_v0(DMControlEnv):
    def grasp_pos(self):
        data = self._dmenv.physics.named.data
        return [data.site_xpos['grasp', 'x'], data.site_xpos['grasp', 'z']]

    def step(self, action):
        _CLOSE = 0.01
        physics = self._dmenv.physics
        old_dist = physics.site_distance('ball', 'target_ball') + \
                   physics.site_distance('ball', 'grasp')
        ts = self._dmenv.step(action)
        obs = flatten_observation(ts.observation)[FLAT_OBSERVATION_KEY]
        info = dict(ts.observation)
        info['ball'] = info['position'][-4:-2]
        info['goal'] = info['target'][:2]
        info['grasp'] = self.grasp_pos()
        info['dist_to_object'] = physics.site_distance('ball', 'grasp')
        info['dist_to_target'] = physics.site_distance('ball', 'target_ball')
        curr_dist = info['dist_to_object'] + info['dist_to_target']

        info['success'] = info['dist_to_object'] < _CLOSE and info[
            'dist_to_target'] < _CLOSE
        done = ts.step_type.last() or info['success']
        reward = ts.reward + old_dist - curr_dist
        info['last'] = ts.step_type.last()
        info['dist_obv'] = curr_dist
        info['reward_survive'] = 0
        info['reward_forward'] = 0
        info['reward_ctrl'] = 0
        info['reward_contact'] = 0
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return

        pixels = self._dmenv.physics.render(width=480, height=480)
        if mode == 'rgb_array':
            return pixels
        elif mode == 'human':
            self.viewer.update(pixels)
        else:
            raise NotImplementedError(mode)
