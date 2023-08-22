"""
使用ray rllib，实现并行训练
"""
import ray
from ray.rllib.agents.ppo import PPOTrainer
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=16, help="number of workers, allow more than the number of physical CPUs")
parser.add_argument("--num-cpus-per-worker", type=float, default=0.1, help="Number of CPUs per worker used.")
parser.add_argument("--num-train", type=int, default=10, help="Number of train.")

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
logger_train = get_logger("train_results", ch_level=logging.DEBUG)

def train_cartpole(num_workers, num_cpus_per_worker, num_train):
    config = {
        "num_workers": num_workers,
        "num_cpus_per_worker": num_cpus_per_worker,
        "framework": "torch",
        "train_batch_size": 4000,
    }
    trainer = PPOTrainer(env="CartPole-v0", config=config)
    for i in range(num_train):
        logger_train.debug(f'--- 开始第{i+1}次训练 ---')
        print(f'--- 开始第{i+1}次训练 ---')
        result = trainer.train()


if __name__ == '__main__':
    # 使用集群训练时用这个
    # ray.init(address="auto")

    # 使用单机训练时用这个
    ray.init()

    args = parser.parse_args()
    logger_train.debug(f'总进程数： {args.num_workers}')
    train_cartpole(args.num_workers, args.num_cpus_per_worker, args.num_train)
