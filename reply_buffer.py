import numpy as np
from gym import spaces


if __name__ == '__main__':
    import gym
    from stable_baselines3.common.buffers import ReplayBuffer
    env = gym.make("CartPole-v1")
    reply_buffer = ReplayBuffer(buffer_size=10,
                                observation_space=env.observation_space,
                                action_space=env.action_space,
                                n_envs=2,
                                handle_timeout_termination = False)
    episodes = 4
    for ep in range(episodes):
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            reply_buffer.add(obs, next_obs, np.array([action]), np.array([reward]), np.array([done]), info)
            obs = next_obs
            if done:
                break
    # print(f'---------- end env, id: {env_id}, buffer: {reply_buffer}')
    print(reply_buffer.observations)
    print(reply_buffer.next_observations)
    print(reply_buffer.actions)
    print(reply_buffer.rewards)
    print(reply_buffer.dones)


