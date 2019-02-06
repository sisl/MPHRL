from PIL import Image
from gym_dmcontrol import *

from config import c
from utils import true_model


class Basic_stack_2_v0(DMControlEnv):

    def __init__(self, domain, task, task_id, task_kwargs=None, visualize_reward=False):
        self._dmenv = suite.load(domain, task,
                                 {'time_limit': (c.bs_per_cpu - 2) / 100,
                                  'task_id': task_id}, visualize_reward)
        self.task = self._dmenv.task
        self.path = self.task.path
        self.physics = self._dmenv.physics
        self.task_id = task_id
        self.num_targets = self.task.num_targets
        self.box_size = self.physics.named.model.geom_size['target0', 0]
        self.num_boxes = 2
        self.stage = []
        self._viewer = None
        self.cost = 10
        self.num_drops = self.task.num_drops
        self.above_box_dist = 4 * self.box_size
        self.num_stages = 6
        self.random_index = 0
        self.one_hot_dim = self.num_stages + c.redundant * 2
        if c.redundant:
            self.stage_vector_size = (self.one_hot_dim + 1) * self.num_boxes
        else:
            self.stage_vector_size = self.one_hot_dim + self.num_boxes
        self.stage_vector_size += 2 * c.repeat
        self.box_of_interests = c.box_of_interest.copy()[self.task_id].tolist()
        self.update_target_loc()
        self.update_stage()
        self.arm = ['upper_arm', 'middle_arm', 'lower_arm']
        self.hand = [
            'hand', 'palm1', 'palm2', 'thumb1', 'thumb2', 'thumbtip1', 'thumbtip2', 'finger1',
            'finger2', 'fingertip1', 'fingertip2'
        ]
        self.stage_color = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [.8, .6, .4, 1], [1, 0, 1, 1],
                            [0, 1, 1, 1]]

    def current_box(self):
        return int(self.box_of_interests[0])

    def update_stage(self):
        if self.is_successful():
            return
        self.old_stage = self.stage.copy()
        self.stage = [0, 0, 0, 0]
        curr_b = self.current_box()
        self.stage[curr_b] = self.get_stage(curr_b)
        if self.stage[curr_b] == -1:
            del self.box_of_interests[0]
            del self.target_loc[0]
            if self.is_successful():
                return
            curr_b = self.current_box()
            self.stage[curr_b] = self.get_stage(curr_b)

        other_b = 1 - curr_b
        self.stage[2 + curr_b] = 1
        self.stage[2 + other_b] = 0
        self.stage[other_b] = -2

    def above(self, box, mult=4):
        return [box[0], box[1] + mult * self.box_size]

    @property
    def observation_space(self):
        obs = self.get_obs()
        return spaces.Box(-np.inf, np.inf, shape=obs.shape)

    def stage_obv(self):
        if c.stage_obv:
            copy = self.stage.copy()
            if c.redundant:
                mat = np.eye(self.one_hot_dim)[[copy[0], copy[1]]]
            else:
                interest = self.current_box()
                mat = np.eye(self.one_hot_dim)[copy[interest]]
            vec = mat.flatten().tolist() + copy[-self.num_boxes:] * (c.repeat + 1)
            assert len(vec) == self.stage_vector_size, (len(vec), self.stage_vector_size)
            return vec
        else:
            return []

    def get_target(self, b):
        if self.is_successful():
            return [0, 0]
        if b in self.box_of_interests:
            index = self.box_of_interests.index(b)
            target = self.target_loc[index]
        else:
            index = 0
            target = self.target_loc[index]
        return target

    def box_to_target_dist(self, calculate=False):
        dist = []
        if c.dist_obv or calculate:
            grasp = self.loc('grasp')
            for b in range(self.num_boxes):
                box = self.loc('box' + str(b))
                target = self.get_target(b)
                dist += [
                    grasp[0] - box[0], grasp[1] - box[1], target[0] - box[0], target[1] - box[1],
                    grasp[0] - target[0], grasp[1] - target[1]
                ]
            box0 = self.loc('box' + str(0))
            box1 = self.loc('box' + str(1))
            dist += [box0[0] - box1[0], box0[1] - box1[1]]
        return dist

    def is_successful(self):
        return self.box_of_interests == []

    def calc_dist_by_box(self, b):
        box = self.loc('box' + str(b))
        grasp = self.loc('grasp')
        stage = self.stage[b]
        target = self.get_target(b)
        above_target = self.above(target, mult=c.above_target)

        if stage == 0:
            # NOTE: REACH
            dist = self.dist(self.above(box), grasp, bounded=c.bounded) * self.cost
        elif stage == 1:
            # Note: LOWER
            dist = self.dist(grasp, box, bounded=c.bounded) * self.cost * 2
        elif stage == 2:
            # NOTE: GRASP
            dist = self.finger_dist() * self.cost * self.cost / 2 - self.dist(
                box, target, bounded=True) * self.cost / 2
        elif stage == 3:
            # NOTE: LIFT
            dist = self.finger_dist() + abs(box[1] - above_target[1]) * self.cost * 2
        elif stage == 4:
            # NOTE: BRING
            dist = self.dist(box, above_target, bounded=c.bounded) * self.cost + self.finger_dist()
        elif stage == 5:
            # NOTE: RELEASE
            dist = self.cost / 2 - self.finger_dist() * self.cost * 2 - \
                   self.dist(grasp, box, bounded=c.bounded) * 2
        else:
            dist = self.dist(self.above(box), grasp, bounded=c.bounded) * self.cost
            # assert False, (
            #     self.box_of_interests, self.stage, b, self.target_loc)

        return dist + (self.num_stages - stage) * self.cost

    def calc_dist(self):
        if self.is_successful():
            return 0

        self.box_of_interest = self.current_box()

        self.dists = []
        self.dists.append(self.calc_dist_by_box(self.box_of_interest))

        for _ in self.box_of_interests[1:]:
            self.dists.append((self.num_stages + 1) * self.cost)

        return sum(self.dists)

    def face_down(self, coeff):
        ori = self.calc_hand_ori()
        return abs(ori[0]) < coeff * self.box_size and ori[1] > 0

    def get_stage(self, b):
        grasp = self.loc('grasp')
        box = self.loc('box' + str(b))
        target = self.get_target(b)
        above_target = self.above(target, mult=c.above_target)
        above = self.above(box)
        midtip = self.midtip()

        # NOTE: IF ON GROUND
        if abs(box[1] - target[1]) < 0.002:
            # NOTE: -1 if BOX IS CLOSE TO TARGET
            if self.dist(box, target) < c.drop_close * self.box_size:
                if abs(grasp[0] - box[0]) > self.box_size or abs(grasp[1] -
                                                                 box[1]) > 1.5 * self.box_size:
                    return -1
                else:
                    return 5
        # NOTE: 1. GRASP is CLOSE TO ABOVE TARGET
        if self.dist(above_target, grasp) < c.bring_close * self.box_size:
            # NOTE: BOX IS FALLING TOWARDS THE TARGET
            if box[1] - target[1] >= -0.002:
                if abs(box[0] - target[0]) < c.drop_width * self.box_size:
                    return 5

        # NOTE: RELEASE
        # NOTE: 1. Box is in grasp
        if self.dist(grasp, box) < 2 * self.box_size and self.face_down(1):
            if self.finger_dist() < 2.5 * self.box_size:
                if abs(box[1] - above_target[1]) < 2 * self.box_size:
                    return 4
                # NOTE: BRING
                return 3
            else:
                # NOTE: GRASP
                return 2
        # NOTE: 1. Tip is below box top
        # NOTE: 2. Grasp is above Tip
        # NOTE: 3. Grasp is open
        if box[1] < midtip[1] < above[1] and abs(
                midtip[0] - above[0]) < 2 * self.box_size \
                and self.face_down(1) \
                and self.finger_dist() > 2 * self.box_size:
            # NOTE: LOWER
            return 1

        # NOTE: REACH
        return 0

    def loc(self, body):
        return np.array(self.physics.named.data.site_xpos[body, ['x', 'z']])

    def reset(self):
        self._dmenv.reset()
        self.random_index = self.task.random_index
        self.box_of_interests = \
            c.box_of_interest.copy()[self.task_id].tolist()[self.random_index:]
        self.update_target_loc()
        self.update_stage()
        self.curr_dist = self.calc_dist()
        return self.get_obs()

    def get_obs(self):
        obs = flatten_observation(
            self._dmenv.task.get_observation(self._dmenv.physics))[FLAT_OBSERVATION_KEY]
        return np.concatenate((obs, self.box_to_target_dist(), self.stage_obv()))

    def update_target_loc(self):
        self.target_loc = []
        for i in range(self.random_index, self.num_targets):
            self.target_loc.append(self.loc('target%s' % i))

    def finger_dist(self):
        thumbtip = self.loc('thumbtip_touch')
        fingertip = self.loc('fingertip_touch')
        return self.dist(thumbtip, fingertip, bounded=False)

    def step(self, action, model):
        old_dist = self.curr_dist
        ts = self._dmenv.step(action)
        self.update_stage()
        self.curr_dist = self.calc_dist()

        info = dict(ts.observation)
        obs = self.get_obs()
        reward = old_dist - self.curr_dist
        done = ts.step_type.last() or self.is_successful()
        # print(old_dist, self.curr_dist, reward, done, self.is_successful(),
        #       self.box_of_interests, self.stage)
        info['success'] = self.is_successful()
        info['last'] = ts.step_type.last()
        info['dist'] = self.curr_dist
        info['finger_dist'] = self.finger_dist()
        info['finger_reward'] = -info['finger_dist'] * self.cost * self.cost
        info['dists'] = self.dists.copy()
        info['target'] = self.target_loc.copy()
        info['stage_obv'] = self.stage.copy()
        info['hand'] = self.calc_hand_ori()
        info['grasp'] = self.loc('grasp').copy()
        info['box_of_interest'] = self.box_of_interest
        info['box_of_interests'] = self.box_of_interests.copy()
        info['grasp_to_abo_tar_dist'] = [
            self.dist(info['grasp'], self.above(i, mult=c.above_target)) for i in info['target']
        ]
        info['box_size'] = self.box_size
        info['reward_survive'] = 0
        info['reward_forward'] = 0
        info['reward_ctrl'] = 0
        info['reward_contact'] = 0
        info['correct_model'] = true_model([info])[0]

        if c.change_color:
            b = self.box_of_interest
            for geom in self.arm:
                self.task.model.geom_rgba[geom] = self.task.model.geom_rgba['box' + str(model // 6)]
            for geom in self.hand:
                self.task.model.geom_rgba[geom] = self.stage_color[model % 6]
            for i in range(self.num_targets):
                target_name = 'target%s' % self.task.box_to_target[b]
                if i == self.task.box_to_target[b]:
                    self.task.model.geom_rgba[target_name][-1] = 0.5
                else:
                    self.task.model.geom_rgba[target_name][-1] = 0.05

        return obs, reward, done, info

    def midtip(self):
        thumbtip = self.loc('thumbtip_touch')
        fingertip = self.loc('fingertip_touch')
        return (thumbtip + fingertip) / 2

    def calc_hand_ori(self):
        thumbtip = self.loc('thumbtip_touch')
        fingertip = self.loc('fingertip_touch')
        mid_point = (thumbtip + fingertip) / 2
        grasp = self.loc('grasp')
        return grasp - mid_point

    def dist_raw(self, obj1, obj2):
        if c.manhattan:
            return np.sum(np.abs([obj1[0] - obj2[0], obj1[1] - obj2[1]]))
        else:
            return np.linalg.norm([obj1[0] - obj2[0], obj1[1] - obj2[1]])

    def dist(self, obj1, obj2, bounded=False):
        if bounded:
            return min(c.dist_bound, self.dist_raw(obj1, obj2))
        else:
            return self.dist_raw(obj1, obj2)

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return

        pixels = self._dmenv.physics.render(width=480, height=480)
        if c.gif_format == 'png' and not c.change_color:
            height = pixels.shape[0]
            width = pixels.shape[1]
            new_width = width * 0.7
            new_height = height * 0.7
            im = Image.fromarray(np.uint8(pixels))
            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            pixels = np.array(im.crop((left, top, right, bottom)))
        if mode == 'rgb_array':
            return pixels
        elif mode == 'human':
            self.viewer.update(pixels)
        else:
            raise NotImplementedError(mode)
