"""
使用python自带的多进程库：multiprocessing，实现并行采样

"""

import multiprocessing
import numpy as np
from typing import Dict, Tuple, Union
import gym
from gym import spaces
from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
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


def get_action_dim(action_space: spaces.Space) -> int:
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

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.infos = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

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



def sample(env_id, reply_buffer):
    print(f'---------- start env, id: {env_id}')
    episode = 200
    env = gym.make("CartPole-v1")

    for ep in range(episode):
        # print(f'env id: {env_id}, episode: {ep}')
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            reply_buffer.add(obs, next_obs, action, reward, done, env_id)
            obs = next_obs
            if done:
                break
    print(f'---------- end env, id: {env_id}')

class MyManager(BaseManager):
    pass


def run_process():
    env = gym.make("CartPole-v1")
    MyManager.register('Buffer', ReplayBuffer)
    with MyManager() as manager:
        # ms = manager.get_server()
        # ms.serve_forever()

        rb = manager.Buffer(buffer_size=1000, observation_space=env.observation_space,action_space=env.action_space)
        mp = [Process(target=sample, args=(i,rb)) for i in range(100)]
        [p.start() for p in mp]
        [p.join() for p in mp]

        # ep = rb.get_eps()
        # print(ep)

if __name__ == '__main__':
    print('start multi process')
    multiprocessing.set_start_method('spawn')   # save memory
    run_process()