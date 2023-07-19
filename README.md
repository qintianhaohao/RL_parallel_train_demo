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


# 参考
<https://docs.ray.io/en/releases-1.13.0/cluster/cloud.html#cluster-cloud>
