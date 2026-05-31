# Mind-Fly1.2

Mind-Fly1.2 是一个基于 MindSpore + PPO 的无人机/移动机器人避障策略项目。项目使用 128 线归一化激光雷达输入和目标方向等任务特征，输出前向速度 `vx` 与角速度 `omega`，并提供本地训练、策略推理、TensorBoard 监控以及 Orange Pi AIpro + PX4 的部署脚本。

## 项目结构

```text
Mind-Fly_1.2/
├── config/                         # 训练环境与 PPO 参数
│   ├── env_config.json
│   └── train_config.json
├── mind_env/                       # 仿真环境、雷达观测和奖励逻辑
├── mind_ppo/                       # PPO 模型、缓存、训练入口
├── deploy/                         # 本地部署辅助代码
├── ms_orangepi_aipro_px4_deploy/   # Orange Pi AIpro + PX4 实机部署
├── mind_infer.py                   # 已训练策略的本地推理/行为探针
├── start_tensorboard.py            # TensorBoard 启动脚本
└── runs/                           # 训练日志和权重输出，默认不提交到 Git
```

## 主要特性

- MindSpore 实现的 PPO 策略网络
- 128 维激光雷达观测 + 7 维任务状态输入
- 输出 `[vx_mps, omega_radps]` 两维控制量
- 支持课程学习、熵控制、KL 早停、梯度裁剪等训练稳定化策略
- 支持 TensorBoard 记录训练指标
- 提供 Orange Pi AIpro、RPLidar、PX4 MAVLink 实机部署参考

## 环境准备

建议使用 Python 虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装常用依赖：

```bash
pip install -U pip
pip install numpy tensorboard tensorboardX
```

然后根据运行设备安装对应版本的 MindSpore。训练配置默认使用 Ascend：

```json
{
  "device": "Ascend"
}
```

如果在 CPU 环境调试，可以把 `config/train_config.json` 中的 `device` 改为 `CPU`。

## 训练

使用默认配置启动训练：

```bash
python mind_ppo/mind_train.py --train_config config/train_config.json
```

训练配置位于：

- `config/train_config.json`：PPO 参数、采样规模、日志与 checkpoint 路径
- `config/env_config.json`：速度限制、仿真步长、安全距离、奖励项和雷达观测参数

默认训练输出会写入 `runs/` 目录，例如：

```text
runs/ppo_exp2/latest_policy.ckpt
runs/ppo_exp2/final_policy.ckpt
runs/ppo_exp2/final_optimizer.ckpt
```

`runs/` 已被 `.gitignore` 排除，避免把训练日志和大模型文件提交到 GitHub。

## TensorBoard

如果训练日志在 `runs/` 下，可以直接运行：

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

也可以按需修改 `start_tensorboard.py` 中的 `--logdir` 后启动：

```bash
python start_tensorboard.py
```

浏览器访问：

```text
http://127.0.0.1:6006
```

## 本地推理

`mind_infer.py` 会加载：

```text
runs/ppo_exp1/latest_policy.ckpt
```

运行行为探针：

```bash
python mind_infer.py
```

如果你的权重在其他目录，请修改 `mind_infer.py` 中的 `CKPT_FILE`，或将权重复制到默认路径。

策略输入输出约定：

```text
obs: [1, 135]
     [128 normalized lidar rays,
      ref_sin, ref_cos,
      prev_vx_n, prev_omega_n,
      dvx_n, domega_n,
      dist_n]

action: [1, 2] = [vx_mps, omega_radps]
```

速度限制来自 `config/env_config.json`：

```json
{
  "vx_max": 0.6,
  "omega_max": 1.5
}
```

## Orange Pi AIpro + PX4 部署

部署相关文件在 `ms_orangepi_aipro_px4_deploy/`，包含：

```text
real_flight_ms.py      实机主循环
flight_control.py      PX4 MAVLink 速度/角速度控制
lidar_reader.py        RPLidar 到 128 线雷达输入转换
runtime_common.py      模型加载、观测构造、安全滤波
export_mindir.py       ckpt 导出 MindIR
benchmark_infer.py     推理速度测试
```

在板端安装部署依赖：

```bash
pip install -r ms_orangepi_aipro_px4_deploy/requirements.txt
```

无硬件冒烟测试：

```bash
python3 ms_orangepi_aipro_px4_deploy/real_flight_ms.py --dry-run
```

连接 RPLidar 和 PX4 后运行：

```bash
python3 ms_orangepi_aipro_px4_deploy/real_flight_ms.py \
  --device-target CPU \
  --lidar-port /dev/ttyUSB0 \
  --px4-port /dev/ttyAMA1 \
  --target-x 10.0 \
  --target-y 0.0
```

更多部署、导出和性能测试说明见：

```text
ms_orangepi_aipro_px4_deploy/README.md
```

## 模型导出

导出 MindIR：

```bash
python3 ms_orangepi_aipro_px4_deploy/export_mindir.py \
  --output ms_orangepi_aipro_px4_deploy/artifacts/ppo_policy
```

转换 OM 模型模板：

```bash
SOC_VERSION=Ascend310B1 bash ms_orangepi_aipro_px4_deploy/convert_om.sh
```

请根据 Orange Pi AIpro 的 CANN 镜像版本调整 `SOC_VERSION`。

## Git 提交说明

本仓库默认不提交以下内容：

- `runs/` 训练日志和 checkpoint
- `__pycache__/`、`.ipynb_checkpoints/` 等缓存
- `*.log`、`*.ckpt`、TensorBoard event 文件

如果需要共享训练权重，建议使用 GitHub Release、网盘或对象存储，并在 README 中注明下载地址和放置路径。

## 注意事项

- 实机飞行前请先完成 `--dry-run`、推理延迟测试和安全距离校准。
- 雷达方向需要与模型约定一致：`ray 0` 前方，`ray 32` 左方，`ray 64` 后方，`ray 96` 右方。
- PX4、RPLidar 和 MindSpore/CANN 版本需要与目标硬件环境匹配。
