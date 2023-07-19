from typing import Dict, Tuple, Union
import numpy as np
import gym
from gym import spaces
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Process, Lock


class MyManager(BaseManager):
    pass



def sample(env_id, reply_buffer):
    print(f'---------- start env, id: {env_id}')
    episode = 2000
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



def run_remote_client():
    # build connect
    MyManager.register('get_buffer')
    manager = MyManager(address=('192.168.202.101', 50000), authkey=b'aaaa')
    manager.connect()
    buffer = manager.get_buffer()
    
    # test connect
    obs = buffer.get_obs()
    print(obs)

    # write something
    mp = [Process(target=sample, args=(i,buffer)) for i in range(30)]
    [p.start() for p in mp]
    [p.join() for p in mp]


def run_remote_client_get_obs():
    # build connect
    MyManager.register('get_buffer')
    manager = MyManager(address=('192.168.202.101', 50000), authkey=b'aaaa')
    manager.connect()
    
    # test connect
    while True:
        obs = buffer.get_obs()
        print(obs)


if __name__ == '__main__':
    print('start multi process')
    # multiprocessing.set_start_method('spawn')   # save memory

    # run_remote_client()
    run_remote_client_get_obs()