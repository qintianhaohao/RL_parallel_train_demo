"""
使用ray自带的多进程库：multiprocessing，和python的线程池，实现并行采样
"""
import gym
import ray
import numpy as np
from ray.util.multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

class CartPoleAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy = self._build_policy()
        self.env = gym.make("CartPole-v1")

    def _build_policy(self):
        # Implement your policy network here (e.g., a simple neural network).
        pass

    def get_action(self, state):
        # Implement the function to get the agent's action based on the current state.
        return self.env.action_space.sample()

    def update_policy(self):
        # Implement the function to update the policy using the collected experiences.
        pass


def run_episode(agent):
    env = gym.make("CartPole-v1")
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    env.close()
    # print('eps 1: ', total_reward)
    return total_reward


def parallel_sampling(args):
    num_episodes = args[0]
    num_threads = args[1]
    agent = CartPoleAgent(state_dim=4, action_dim=2) # Change the dimensions based on the CartPole environment.

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        thread_results = executor.map(run_episode, [agent] * num_episodes)
        for i, results in enumerate(thread_results):
            print(i, results)


def run_ray_pool(num_episodes_per_thread, num_threads, num_processes):
    """

    :return:
    """
    args = [(num_episodes_per_thread, num_threads) for _ in range(num_processes)]
    pool = Pool(processes=num_processes)
    for i, result in enumerate(pool.map(parallel_sampling, args)):
        print('process:', i, result)


if __name__ == "__main__":
    num_episodes_per_thread = 5
    num_threads = 2
    num_processes = 8

    # parallel_sampling(num_episodes_per_thread, num_threads)
    run_ray_pool(num_episodes_per_thread, num_threads, num_processes)