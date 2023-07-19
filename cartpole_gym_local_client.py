import time
from typing import Dict, Tuple, Union
import numpy as np
import gym
from gym import spaces
import multiprocessing
from multiprocessing.managers import BaseManager
from multiprocessing import Process, Lock


def sample(env_id, reply_buffer):
    print(f'---------- start env, id: {env_id}')
    episode = 2000
    env = gym.make("CartPole-v1")

    for ep in range(episode):
        # print(f'env id: {env_id}, episode: {ep}')
        obs = env.reset()
        while True:
            # time.sleep(1)
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            reply_buffer.add(obs, next_obs, action, reward, done, env_id)
            obs = next_obs
            if done:
                break
    print(f'---------- end env, id: {env_id}')


def put_queue(env_id, queue):
    print(f'---------- start env, id: {env_id}')
    for i in range(200000):
        queue.put({env_id: i})
    print(f'---------- end env, id: {env_id}')



class MyManager(BaseManager):
    pass

def run_local_client():
    # build connect
    MyManager.register('get_buffer')
    manager = MyManager(address=('192.168.1.4', 50000), authkey=b'aaaa')
    manager.connect()
    buffer = manager.get_buffer()

    # write something
    mp = [Process(target=sample, args=(i,buffer)) for i in range(200)]
    [p.start() for p in mp]
    [p.join() for p in mp]

    # MyManager.register('get_queue')
    # manager = MyManager(address=('192.168.1.4', 50000), authkey=b'aaaa')
    # manager.connect()
    # q = manager.get_queue()
    # mp = [Process(target=put_queue, args=(i,q)) for i in range(10)]
    # [p.start() for p in mp]
    # [p.join() for p in mp]



if __name__ == '__main__':
    print('start multi process')
    # multiprocessing.set_start_method('spawn')   # save memory

    run_local_client()