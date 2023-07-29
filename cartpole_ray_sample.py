"""
使用ray.remote，实现并行采样
"""
import gym
import ray
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=16, help="number of workers, allow more than the number of physical CPUs")
parser.add_argument("--num-episodes-per-worker", type=int, default=2000, help="Number of episodes per worker.")


# @ray.remote(num_cpus=0.5)
@ray.remote
def cartpole_sample(args):
    env_id = args[0]
    reply_buffer = ray.get(args[1])
    episodes = args[2]
    # print(f'reply buffer: {reply_buffer}')

    print(f'--- 启动第 {env_id} 个采样环境 ---')
    env = gym.make("CartPole-v1")
    reply_buffer = ReplayBuffer(buffer_size=10,
                                observation_space=env.observation_space,
                                action_space=env.action_space,
                                n_envs=1,
                                handle_timeout_termination = False)
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            # reply_buffer.add(obs, next_obs, action, reward, done, env_id)
            # reply_buffer[env_id] = env_id
            # reply_buffer.add(obs, next_obs, np.array([action]), np.array([reward]), np.array([done]), info)
            obs = next_obs
            if done:
                break
    # print(f'---------- end env, id: {env_id}, buffer: {reply_buffer}')
    print(f'--- 第 {env_id} 个环境完成采样 ---')

    return reply_buffer


def run_ray_pool(num_workers, num_episodes_per_worker):
    """
    :return:
    """
    # buffer = multiprocessing.Array('i', 3)
    # buffer = array.array('i', [0]*3)
    buffer_id = ray.put([0] * num_workers)
    # buffer_id.flags.writeable = True
    # buffer_id = [0]*3
    # args = [(i, buffer_id) for i in range(10)]

    buffer = ray.get([cartpole_sample.remote([i, buffer_id, num_episodes_per_worker]) for i in range(num_workers)])
    print('buffer: ', buffer)


if __name__ == '__main__':
    # this can be used on an already started ray cluster
    # ray.init(address="auto")

    # this can be used on a single node
    ray.init()

    args = parser.parse_args()
    print(f'总进程数： {args.num_workers}')
    run_ray_pool(args.num_workers, args.num_episodes_per_worker)