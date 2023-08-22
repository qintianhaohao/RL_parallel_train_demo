"""
使用ray.remote，实现并行采样
"""
import gym
import ray
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=16, help="number of workers, allow more than the number of physical CPUs")
parser.add_argument("--num-episodes-per-worker", type=int, default=2000, help="Number of episodes per worker.")

# 使用logger输出
def get_logger(name="", print_cmd=True, ch_level=logging.WARNING):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 控制台输出等级
    if print_cmd:
        ch = logging.StreamHandler()
        ch.setLevel(ch_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
logger_sample = get_logger("sample_results", ch_level=logging.DEBUG)


# @ray.remote(num_cpus=0.5) # 可以设置单个worker的cpu数，但是不设置也不影响运行
@ray.remote
def cartpole_sample(args):
    env_id = args[0]
    buffer_id = ray.get(args[1])
    episodes = args[2]
    # print(f'reply buffer: {reply_buffer}')

    logger_sample.debug(f'--- 启动第{env_id+1}个采样环境 ---')
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
    print(f'--- 第{env_id+1}个环境完成采样 ---')
    logger_sample.debug(f'--- 第{env_id+1}个环境完成采样 ---')
    # logger_sample.debug(f'buffer: {reply_buffer}')
    return reply_buffer

# todo: 调试reply_buffer
def run_ray_pool(num_workers, num_episodes_per_worker):
    """
    使用进程池
    :return:
    """
    buffer_id = ray.put([0] * num_workers)  # 这里其实应该是reply_buffer
    buffer = ray.get([cartpole_sample.remote([i, buffer_id, num_episodes_per_worker]) for i in range(num_workers)])
    # print('buffer: ', buffer)   # 最后再拿到reply_buffer


if __name__ == '__main__':
    # 使用集群训练时用这个
    # ray.init(address="auto")

    # 使用单机训练时用这个
    ray.init()

    args = parser.parse_args()
    logger_sample.debug(f'总进程数： {args.num_workers}')
    run_ray_pool(args.num_workers, args.num_episodes_per_worker)