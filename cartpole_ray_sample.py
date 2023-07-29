"""
使用ray，实现并行采样
"""
import gym
import ray


@ray.remote(num_cpus=0.5)
def cartpole_sample(args):
    env_id = args[0]
    reply_buffer = ray.get(args[1])
    episodes = args[2]
    print(f'reply buffer: {reply_buffer}')

    print(f'----- start env, id: {env_id}')
    env = gym.make("CartPole-v1")

    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            # reply_buffer.add(obs, next_obs, action, reward, done, env_id)
            # reply_buffer[env_id] = env_id
            obs = next_obs
            if done:
                break
    print(f'---------- end env, id: {env_id}, buffer: {reply_buffer}')

    return reply_buffer


def run_ray_pool(num_episodes_per_worker, num_workers):
    """

    :return:
    """

    # buffer = multiprocessing.Array('i', 3)
    # buffer = array.array('i', [0]*3)
    buffer_id = ray.put([0] * num_workers)
    # buffer_id.flags.writeable = True
    # buffer_id = [0]*3
    # args = [(i, buffer_id) for i in range(10)]

    print(ray.get([cartpole_sample.remote([i, buffer_id, num_episodes_per_worker]) for i in range(num_workers)]))



if __name__ == '__main__':
    num_episodes_per_worker = 20000
    num_workers = 8

    ray.init()
    
    run_ray_pool(num_episodes_per_worker, num_workers)