# 注意
本项目所使用的软件和包的版本如下：
python: 3.10
ray: 1.13.0

# 如何启动ray集群
## 手动启动
在集群主节点，输入：
```cmd
ray start --head --port=6379
```

在集群其他节点，输入下面命令，把`<address>`换成节点地址，如：`123.45.67.89:6379`,
```cmd
ray start --address=<address>
```

如果要中止某个节点，直接在该节点运行：
```cmd
ray stop
```

# 如何使用ray集群进行训练
## python
只需要在训练代码里，加入：
```python
ray.init(address="auto")
```
示例代码：
```python
from collections import Counter
import socket
import time

import ray

ray.init()

print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

@ray.remote
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))
```

# 并行采样、并行训练测试
## 切换conda环境
```commandline
conda activate py310
```

## 并行采样
```commandline
python3 cartpole_ray_sample.py --num-workers 50 --num-episodes-per-worker 1000
```

## 并行训练
```commandline
python3 cartpole_ray_train.py --num-workers 80 --num-cpus-per-worker 0.1
```

# 参考
## ray搭建集群
<https://docs.ray.io/en/releases-1.13.0/cluster/cloud.html#cluster-cloud>
<https://docs.ray.io/en/releases-1.13.0/cluster/guide.html?highlight=8265#deploying-an-application>

## ray.remote 并行计算
<https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resource-requirements>
<https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html>
