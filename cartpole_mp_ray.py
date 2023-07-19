"""
使用ray自带的多进程库：multiprocessing，实现并行采样
"""
import gym
import ray
from ray.util.multiprocessing import Pool



def cartpole_sample(args):
    env_id = args[0]
    reply_buffer = ray.get(args[1])

    print(f'----- start env, id: {env_id}')
    episode = 200000
    env = gym.make("CartPole-v1")

    for ep in range(episode):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            # reply_buffer.add(obs, next_obs, action, reward, done, env_id)
            reply_buffer[env_id] = env_id
            obs = next_obs
            if done:
                break
    print(f'---------- end env, id: {env_id}')
    return reply_buffer


def run_ray_pool():
    """

    :return:
    """

    # buffer = multiprocessing.Array('i', 3)
    # buffer = array.array('i', [0]*3)
    buffer_id = ray.put([0]*10)
    args = [(i, buffer_id) for i in range(10)]

    pool = Pool(processes=8)

    # method 1: use map()
    for result in pool.map(cartpole_sample, args):
        print(result)

if __name__ == '__main__':
    run_ray_pool()