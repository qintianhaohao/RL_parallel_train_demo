"""
使用ray rllib，实现并行训练
"""
import ray
from ray.rllib.agents.ppo import PPOTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=16, help="number of workers, allow more than the number of physical CPUs")
parser.add_argument("--num-cpus-per-worker", type=float, default=0.1, help="Number of CPUs per worker used.")


def train_cartpole(num_workers, num_cpus_per_worker):
    config = {
        "num_workers": num_workers,
        "num_cpus_per_worker": num_cpus_per_worker,
        "framework": "torch",
        "train_batch_size": 4000,
    }
    trainer = PPOTrainer(env="CartPole-v0", config=config)
    for i in range(100):
        print(f'--- 开始第 {i} 次训练 ---')
        print(trainer.train())


if __name__ == '__main__':
    # ray.init(address="auto")

    # this can be used on a single node
    ray.init()

    args = parser.parse_args()
    print(f'总进程数： {args.num_workers}')
    train_cartpole(args.num_workers, args.num_cpus_per_worker)
