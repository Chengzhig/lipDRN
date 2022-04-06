import random
import tensorflow as tf


class MnEnviroment(object):
    def __init__(self, x, y):
        self.train_X = x
        #self.train_Y = y
        self.train_Y = y.numpy().tolist()
        self.current_index = self._sample_index()
        self.action_space = len(set(y)) - 1

    def reset(self):
        obs, _ = self.step(-1)
        return obs

    '''
    action: 0-9 categori, -1 : start and no reward
    return: next_state(image), reward
    '''

    def step(self, action):
        if action == -1:
            _c_index = self.current_index
            self.current_index = self._sample_index()
            return (self.train_X[_c_index], 0)
        r = self.reward(action)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index], r

    def reward(self, action):
        c = self.train_Y[self.current_index]
        # print(c)
        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space)

    def _sample_index(self):
        return  random.randint(0, len(self.train_Y) - 1)
