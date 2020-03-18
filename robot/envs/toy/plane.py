import copy
from collections import OrderedDict

import numpy as np
import gym
import cv2


def line_intersection(line1, line2):
    #calculate the intersection point
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0]
             [1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def check_cross(x0, y0, x1, y1):
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    return np.cross(x1-x0, y0-x0), np.cross(y0-x0, y1-x0)


def check_itersection(x0, y0, x1, y1):
    EPS = 1e-10

    def sign(x):
        if x > EPS:
            return 1
        if x < -EPS:
            return -1
        return 0

    f1, f2 = check_cross(x0, y0, x1, y1)
    f3, f4 = check_cross(x1, y1, x0, y0)
    if sign(f1) == sign(f2) and sign(f3) == sign(f4) and sign(f1) != 0 and sign(f3) != 0:
        return True
    return False


class PlaneBase(gym.Env):
    def __init__(self, rects, R, size=512):
        self.rects = rects
        self.n = len(self.rects)
        self.size = size
        self.map = np.ones((size, size, 3), dtype=np.uint8) * 255
        self.R = R
        self.R2 = R ** 2
        self.board = np.array(
            [[0, 0],
             [1, 1]],
            dtype='float32')

        self.action_space = gym.spaces.Box(
            low=-R, high=R, shape=(2,), dtype='float32')
        self.observation_space = gym.spaces.Box(
            low=0., high=1., shape=(2,), dtype='float32')


        for i in range(self.n):
            for j in range(i+1, self.n):
                if check_itersection(self.rects[i][0], self.rects[i][1], self.rects[j][0], self.rects[j][0]):
                    raise Exception("Rectangle interaction with each other")


        for ((x0, y0), (x1, y1)) in rects:
            x0, y0 = int(x0 * size), int(y0 * size)
            x1, y1 = int(x1 * size), int(y1 * size)
            cv2.rectangle(self.map, (x0, y0), (x1, y1), (0, 255, 0), 1)

            ps = np.array([
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ], dtype=np.int32)
            cv2.fillConvexPoly(self.map, ps, (127, 127, 127))

        self.state = (0, 0)
        self.reset()

    def restore(self, obs):
        self.state = (float(obs[0]), float(obs[1]))

    def rect_lines(self, rect):
        (x0, y0), (x1, y1) = rect
        yield (x0, y0), (x1, y0)
        yield (x1, y0), (x1, y1)
        yield (x1, y1), (x0, y1)
        yield (x0, y1), (x0, y0)

    def l2dist(self, x, y):
        return ((y[0] - x[0]) ** 2) + ((y[1] - x[1]) ** 2)

    def check_inside(self, p):
        EPS = 1e-10
        for i in self.rects:
            if p[0] > i[0][0]+EPS and p[0] < i[1][0]-EPS and p[1] > i[0][1]+EPS and p[1] < i[1][1]-EPS:
                return True
        return False

    def step(self, action):
        dx, dy = action
        l = 0.0001
        p = (self.state[0] + dx * l, self.state[1] + dy * l)
        if self.check_inside(p) or p[0] > 1 or p[1] > 1 or p[0] < 0 or p[1] < 0:
            return np.array(self.state), 0, False, {}

        dest = (self.state[0] + dx, self.state[1] + dy)

        md = self.l2dist(self.state, dest)

        _dest = dest
        line = (self.state, dest)

        for i in list(self.rects) + [self.board]:
            for l in self.rect_lines(i):
                if check_itersection(self.state, dest, l[0], l[1]):
                    inter_point = line_intersection(line, l)
                    d = self.l2dist(self.state, inter_point)
                    if d < md:
                        md = d
                        _dest = inter_point

        _dest = (max(min(_dest[0], 1), 0), max(min(_dest[1], 1), 0))
        self.restore(_dest)
        return np.array(self.state), -md, False, {}

    def render(self, mode='human'):
        image = self.map.copy()
        image[:5,:] = 0
        image[-5:,:] = 0
        image[:,:5] = 0
        image[:,-5:] = 0
        x, y = self.state
        x = int(x * self.size)
        y = int(y * self.size)
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(2)
        else:
            return image

    def reset(self):
        inside_rect = True
        while inside_rect:
            a, b = np.random.random(), np.random.random()
            inside_rect = self.check_inside((a, b))

        self.state = np.array((a, b))
        return self.state.copy()


class GoalPlane(gym.Env):
    def __init__(self, rects=[], maze_size=16., action_size=1., distance=1, start=None, goals=None):
        super(GoalPlane, self).__init__()
        self.env = PlaneBase(rects= rects, R=1, size=512)

        self.maze_size = maze_size
        self.action_size = action_size

        self.action_space = gym.spaces.Box(
            low=-action_size, high=action_size, shape=(2,), dtype='float32')

        ob_space = gym.spaces.Box(
            low=0., high=maze_size, shape=(2,), dtype='float32')
        goal_space = ob_space

        self.distance = distance
        self.goals = goals
        self.start = start

        self.goal_space = goal_space
        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space,
        }))
        self.goal = None
        self.reset()

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return reward

    def step(self, action):
        assert self.goal is not None
        action = np.minimum(action, self.action_size)
        action = np.maximum(action, -self.action_size)

        observation, reward, done, info = self.env.step(np.array(action)/self.maze_size) # normalize action
        observation = np.array(observation) * self.maze_size

        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': observation}

        reward = -np.linalg.norm(observation - self.goal, axis=-1)
        info['is_success'] = (reward > -self.distance)
        return out, reward, done, info

    def reset_xy(self, xy, goal):
        self.start = xy
        self.goals = goal

    def reset(self, xy=None, goal=None):
        if self.start is not None:
            self.env.reset()
            observation = np.array(self.start)
            self.env.restore(observation/self.maze_size)
        else:
            observation = self.env.reset() * self.maze_size
        if self.goals is None:
            condition = True
            while condition: # note: goal should not be in the block
                self.goal = self.goal_space.sample()
                condition = self.env.check_inside(self.goal/self.maze_size)
        else:
            self.goal = np.array(self.goals)

        goal = self.goal

        out = {'observation': observation, 'desired_goal': goal}
        out['achieved_goal'] = observation
        return out

    def render(self, mode='human'):
        image = self.env.render(mode='rgb_array')
        goal_loc = copy.copy(self.goal)
        goal_loc[0] = goal_loc[0] / self.maze_size * image.shape[1]
        goal_loc[1] = goal_loc[1] / self.maze_size * image.shape[0]
        cv2.circle(image, (int(goal_loc[0]), int(goal_loc[1])), 20, (255, 0, 0), -1)

        if mode == 'human':
            cv2.imshow('image', image)
            cv2.waitKey(2)
        else:
            return image

    def render_obs(self, state):
        tmp = np.array(self.env.state).copy()
        self.env.state = state['observation']/self.maze_size

        image = self.render(mode='rgb_array')

        self.env.state = tmp
        return image


if __name__ == '__main__':
    env = GoalPlane()
    env.reset()

    while True:
        a = env.action_space.sample()
        env.render()
        _, reward, done, _ = env.step(a)
        print(reward)
        if done:
            env.reset()
