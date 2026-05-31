# PPO 模型指标计算方法

本文档用于说明本项目 PPO 策略网络的模型规模、部署性能和任务效果指标如何计算，便于在论文、答辩 PPT 和实验报告中统一口径。

## 1. 模型基础信息

本项目使用单线 2D 激光雷达，每帧 360 度扫描被重采样为 128 个角度距离点；同时拼接 7 维状态量，作为策略网络输入。

对应代码位置：

- `orangepi_aipro_deploy/runtime_common.py`
- `config/env_config.json`
- `mind_ppo/mind_models.py`
- `mind_ppo/mind_encoder.py`

关键配置：

```python
OBS_RAYS = 128
POSE_DIM = 7
OBS_DIM = OBS_RAYS + POSE_DIM
ACTION_DIM = 2
```

因此：

```text
输入维度 = 128 + 7 = 135
输出维度 = 2
```

其中输出动作维度对应：

```text
vx, omega
```

即线速度和角速度。

## 2. 参数量 Params

参数量表示模型中所有参数张量的元素个数之和。

计算公式：

```text
Params = sum(每个参数张量的元素个数)
```

例如全连接层：

```text
Dense(in_dim, out_dim)
权重参数 = in_dim * out_dim
偏置参数 = out_dim
总参数 = in_dim * out_dim + out_dim
```

本项目当前策略网络配置为：

```python
PPOPolicy(
    vec_dim=135,
    action_dim=2,
    hidden=64,
    d_model=128,
    num_queries=4,
    num_heads=4,
    learnable_queries=True,
)
```

按网络结构统计：

```text
Encoder 参数量：226,624
Policy / Value heads 参数量：66,565
总参数量：293,189
可训练参数量：292,677
非训练参数量：512
```

非训练参数主要来自 BatchNorm 的 moving mean 和 moving variance。

## 3. 权重理论大小 Model Size

模型权重理论大小由参数量和参数精度决定。

计算公式：

```text
权重大小 = 参数量 * 单个参数字节数
```

常见精度：

```text
FP32: 4 bytes / parameter
FP16: 2 bytes / parameter
BF16: 2 bytes / parameter
INT8: 1 byte / parameter
```

本项目模型参数量为 293,189。

FP32 权重大小：

```text
293,189 * 4 = 1,172,756 bytes
1,172,756 / 1024 / 1024 = 1.12 MiB
```

FP16 / BF16 权重大小：

```text
293,189 * 2 = 586,378 bytes
586,378 / 1024 = 572.6 KiB
```

可在报告中表述为：

```text
模型参数量约 0.293M，FP32 权重理论占用约 1.12 MiB。
```

## 4. Checkpoint 文件大小

checkpoint 文件大小是实际保存到磁盘后的权重文件大小。它通常略大于理论权重大小，因为文件中还包含参数名、shape、dtype 等元数据。

PowerShell 查看方法：

```powershell
Get-Item runs\ppo_exp1\latest_policy.ckpt | Select-Object FullName,Length
```

当前项目 checkpoint：

```text
runs/ppo_exp1/latest_policy.ckpt = 1,176,911 bytes
```

换算：

```text
1,176,911 / 1024 / 1024 = 1.12 MiB
```

该值与 FP32 理论大小 1,172,756 bytes 基本一致。

## 5. 参数量统计脚本

如果运行环境已安装 MindSpore，可使用如下脚本复现参数量统计。

```python
import os
import sys
import numpy as np
import mindspore as ms
from mindspore import context

ROOT = r"C:\Users\Administrator\Desktop\华为ict大赛\Mind-Fly_1.2"
sys.path.insert(0, os.path.join(ROOT, "mind_ppo"))

from mind_models import PPOPolicy

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

model = PPOPolicy(
    vec_dim=135,
    action_dim=2,
    hidden=64,
    d_model=128,
    num_queries=4,
    num_heads=4,
    learnable_queries=True,
)

total = 0
trainable = 0

for p in model.get_parameters():
    n = int(np.prod(p.shape))
    total += n
    if p.requires_grad:
        trainable += n
    print(p.name, p.shape, n, p.requires_grad)

print("Total params:", total)
print("Trainable params:", trainable)
print("FP32 size bytes:", total * 4)
print("FP32 size MiB:", total * 4 / 1024 / 1024)
```

期望输出：

```text
Total params: 293189
Trainable params: 292677
FP32 size bytes: 1172756
FP32 size MiB: 1.1184
```

## 6. FLOPs / MACs 计算方法

FLOPs 表示浮点运算次数，MACs 表示乘加次数。很多报告中会使用：

```text
1 MAC ~= 2 FLOPs
```

因为一次乘加包含一次乘法和一次加法。

常见层的 MACs 计算方法如下。

全连接层：

```text
MACs = 输入维度 * 输出维度
```

Conv1d 层：

```text
MACs = 输出长度 * 输出通道数 * kernel_size * 输入通道数 / groups
```

多头注意力中的 QK 计算：

```text
MACs_QK = num_heads * num_queries * num_tokens * head_dim
```

多头注意力中的 AV 计算：

```text
MACs_AV = num_heads * num_queries * num_tokens * head_dim
```

因此注意力部分近似为：

```text
MACs_attention = MACs_QK + MACs_AV
```

本项目中：

```text
num_heads = 4
num_queries = 4
num_tokens = 128
d_model = 128
head_dim = 128 / 4 = 32
```

注意力核心计算量近似：

```text
QK MACs = 4 * 4 * 128 * 32 = 65,536
AV MACs = 4 * 4 * 128 * 32 = 65,536
Attention MACs ~= 131,072
Attention FLOPs ~= 262,144
```

实际总 FLOPs 还需要加上卷积层、SE 模块、MLP、策略头和价值头。

## 7. 推理延迟 Latency

推理延迟表示单次输入到输出动作所需的时间。边缘部署中应在目标设备上实测，例如 OrangePi AIPro 或 Ascend 环境。

计算公式：

```text
Latency = 总推理耗时 / 推理次数
```

测试时建议：

- 先 warmup 50 到 100 次
- 再正式统计 1000 次以上
- 若使用异步设备，需要强制同步或取出结果

示例脚本：

```python
import time
import numpy as np
import mindspore as ms
from mindspore import Tensor

dummy = Tensor(np.zeros((1, 135), dtype=np.float32), ms.float32)

for _ in range(50):
    _ = model(dummy)

N = 1000
t0 = time.perf_counter()

for _ in range(N):
    out = model(dummy)
    _ = out.asnumpy()

t1 = time.perf_counter()

latency_ms = (t1 - t0) / N * 1000
fps = 1000 / latency_ms

print("Latency ms:", latency_ms)
print("FPS:", fps)
```

## 8. FPS / 控制频率

FPS 表示每秒推理次数。对于控制任务，也可以称为控制频率 Hz。

计算公式：

```text
FPS = 1000 / Latency(ms)
Hz = 1 / Latency(s)
```

例如：

```text
若单次推理延迟为 2 ms
FPS = 1000 / 2 = 500 FPS
控制频率约 500 Hz
```

实际控制频率还会受到雷达采样频率、串口通信、动作下发周期和安全滤波等模块影响。

## 9. 内存占用 Memory

内存占用可分为两种口径。

权重内存：

```text
参数量 * 单个参数字节数
```

运行时内存：

```text
权重内存 + 中间激活内存 + 框架运行开销 + 输入输出缓存
```

本项目 FP32 权重内存约为：

```text
1.12 MiB
```

运行时内存需要在目标设备上实测。

CPU 端可使用如下方式粗略查看进程内存：

```python
import os
import psutil

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / 1024 / 1024
print("RSS memory MB:", mem_mb)
```

在 Ascend 或 OrangePi AIPro 上，建议使用对应设备监控工具统计 NPU / 系统内存占用。

## 10. 任务效果指标

导航任务不能只看模型大小，还需要评估策略实际效果。

常用指标如下：

| 指标 | 计算方法 |
| --- | --- |
| 成功率 | 成功到达目标次数 / 测试总次数 |
| 碰撞率 | 碰撞次数 / 测试总次数 |
| 平均奖励 | 所有 episode reward 的平均值 |
| 平均到达时间 | 成功 episode 的耗时平均值 |
| 平均路径长度 | 每轮轨迹长度求平均 |
| 轨迹平滑度 | 速度变化量或角速度变化量的平均值 |

计算公式：

```text
Success Rate = N_success / N_total
Collision Rate = N_collision / N_total
Average Reward = sum(reward_episode) / N_total
Average Arrival Time = sum(time_success) / N_success
```

轨迹平滑度可用动作变化量衡量：

```text
Smoothness_v = mean(abs(vx_t - vx_{t-1}))
Smoothness_omega = mean(abs(omega_t - omega_{t-1}))
```

数值越小，说明控制输出越平滑。

## 11. 和 ResNet / YOLO 的对比口径

本项目 PPO 模型：

```text
参数量：0.293M
FP32 权重大小：1.12 MiB
输入：135 维结构化状态
输出：2 维连续控制动作
```

典型视觉模型：

```text
ResNet-18 10 分类：约 11.18M 参数
YOLOv8n：约 3.2M 参数
YOLO11n：约 2.6M 参数
```

对比关系：

```text
ResNet-18 10 分类约为本项目模型的 38 倍
YOLOv8n 约为本项目模型的 11 倍
YOLO11n 约为本项目模型的 9 倍
```

原因是视觉模型需要从高维图像像素中学习边缘、纹理、形状和语义特征，而本项目 PPO 策略网络输入为低维结构化激光和状态信息，只需学习状态到动作的控制映射。

## 12. 推荐答辩表述

可在 PPT 中这样描述：

```text
本项目 PPO 策略网络输入为 128 维单线激光雷达距离采样点与 7 维状态量，共 135 维；输出为线速度 vx 和角速度 omega 两个连续控制量。当前模型参数量约 0.293M，FP32 权重理论占用约 1.12 MiB，checkpoint 文件大小约 1.12 MiB，属于 Tiny 级轻量化边缘智能模型。相比 ResNet-18 十分类模型约 11.18M 参数，以及 YOLO-nano 系列百万级参数规模，本项目模型更适合部署在资源受限的边缘计算设备上进行实时导航控制。
```

完整指标体系可概括为：

```text
模型规模：Params、Model Size、Checkpoint Size
计算复杂度：FLOPs / MACs
部署性能：Latency、FPS / Hz、Runtime Memory
任务效果：Success Rate、Collision Rate、Average Reward、Arrival Time、Path Smoothness
```
