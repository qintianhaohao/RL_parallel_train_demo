"""
使用ray rllib，实现并行训练
"""
import ray
from ray.rllib.agents.ppo import PPOTrainer



if __name__ == '__main__':
    ray.init(address="auto")

    # run_ray_pool()
    config = {
        "num_workers": 1000,
        # "num_envs_per_worker": 50,
        "num_cpus_per_worker": 0.05,

        "framework": "torch",
        "train_batch_size": 4000,
    }
    trainer = PPOTrainer(env="CartPole-v0", config=config)
    while True:
        print(trainer.train())
