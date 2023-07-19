import time
from typing import Dict, Tuple, Union
import numpy as np
import gym
import queue
from gym import spaces
from multiprocessing.managers import BaseManager



def get_obs_shape(
    observation_space) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.buffer_size = buffer_size
        self.n_envs = 1
        self.pos = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)

        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.infos = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        self.eps = np.zeros((self.buffer_size, 1, 2), dtype=np.float32)


    def add(self, obs, next_obs, action, reward, done, info):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.infos[self.pos] = np.array(info).copy()
        self.eps[self.pos] = np.array([info, done]).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_dones(self):
        return self.dones

    def get_infos(self):
        return self.infos

    def get_eps(self):
        return self.eps

    def get_obs(self):
        return self.observations

class MyManager(BaseManager):
    pass


env = gym.make("CartPole-v1")
my_buffer = ReplayBuffer(buffer_size=20, observation_space=env.observation_space, action_space=env.action_space)
def get_buffer():
    return my_buffer


my_q = queue.Queue()
def get_queue():
    return my_q


def run_process_server():

    MyManager.register('get_buffer', callable=get_buffer)
    manager = MyManager(address=('', 50000), authkey=b'aaaa')

    # ms = manager.get_server()
    # ms.serve_forever()

    manager.start()
    buffer = manager.get_buffer()
    while True:
        # time.sleep(1)
        env_id = buffer.get_infos()
        print('env_id: ', env_id.tolist())

    # MyManager.register('get_queue', callable=get_queue)
    # manager = MyManager(address=('', 50000), authkey=b'aaaa')
    # manager.start()
    # q = manager.get_queue()
    # while True:
    #     r = q.get(timeout=100)
    #     print(r)


if __name__ == '__main__':
    print('start multi process')
    # multiprocessing.set_start_method('spawn')   # save memory

    run_process_server()